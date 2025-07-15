import ast
import copy
import cornac
import itertools
import numpy as np
import optuna
import os
import polars as pl
import recbole
import recbole.config
import recbole.data
import recbole.data.interaction
import recbole.model.general_recommender
import recbole.trainer
import warnings
import yaml

from datafactory import load_lbsn_artefacts
from utils import argnonz, argtopk, rand_member_nb, rand_member_arr_nb, myrecall


warnings.filterwarnings("ignore", category=FutureWarning)  # pandas warning


TOPK = 20
HIDDEN_SIZE = 32


def formatdict(d):
    s = " ".join(f"{k}={v:>7.4f}" if not isinstance(v, str) else f"{k}={v}" for k, v in d.items())
    s = "{" + s + "}"
    return s


# ----------------------------------- CORNAC ----------------------------------


def isOK(y, uid_map, iid_map):
    assert (
        1
        and set(np.unique(y)) == {0,1}
        and y.shape[0] == len(uid_map)
        and y.shape[1] == len(iid_map)
        and list(uid_map.keys()) == list(uid_map.values())
        and list(iid_map.keys()) == list(iid_map.values())
    )
    return True


def prepare_positive_feedback_data(y, uid_map, iid_map, seed):  # --> only positive feedback
    assert isOK(y, uid_map, iid_map)
    N, J = y.shape
    n_implicit_feedback = y.sum()
    uir_tup = (
        np.zeros(n_implicit_feedback, dtype="i8"),  # user_indices
        np.zeros(n_implicit_feedback, dtype="i8"),  # item_indices
        np.zeros(n_implicit_feedback, dtype="f8"),  # feedback values (ratings)
    )
    counter = 0
    for n in range(N):
        for j in range(J):
            if y[n,j] != 0:
                uir_tup[0][counter] = uid_map[n]
                uir_tup[1][counter] = iid_map[j]
                uir_tup[2][counter] = 1
                counter += 1
    assert n_implicit_feedback == counter, "Implicit feedback mismatch detected"
    data = cornac.data.Dataset(num_users=N, num_items=J, uid_map=uid_map, iid_map=iid_map, uir_tuple=uir_tup, seed=seed)
    return data


def prepare_and_split_positive_feedback_data(y, uid_map, iid_map, seed, te_frac=0.33):  # --> w\i user train-test items split
    assert isOK(y, uid_map, iid_map) and (0 < te_frac < 1)
    N = len(y)
    tr_y = np.zeros(y.shape, dtype="i8")
    te_y = np.zeros(y.shape, dtype="i8")
    tr_frac = 1 - te_frac
    for n in range(N):
        pos = argnonz(y[n,:])
        tr_pos = rand_member_arr_nb(pos, size=max(1, int(len(pos) * tr_frac)))
        te_pos = np.setdiff1d(pos, tr_pos)
        tr_y[n,tr_pos] = 1
        te_y[n,te_pos] = 1
    return (
        prepare_positive_feedback_data(tr_y, uid_map, iid_map, seed),
        prepare_positive_feedback_data(te_y, uid_map, iid_map, seed),
    )


# -----------------------------------------------------------------------------


def hsearch(
        model: cornac.models.Recommender, space: dict[str,np.ndarray|list], space_tied_constraints: dict[str,str],
        tr_data: cornac.data.Dataset,
        te_data: cornac.data.Dataset,
        gridsearch: bool = True, n_iters: int = 0,
):

    assert (tr_data.num_users == te_data.num_users) and (tr_data.uid_map == te_data.uid_map), "Shape or Map mismatch"
    assert (tr_data.num_items == te_data.num_items) and (tr_data.iid_map == te_data.iid_map), "Shape or Map mismatch"

    # grid search hyperparameters optimisation or not?
    __inp = copy.deepcopy(space)
    if gridsearch:
        space = (dict(zip(__inp.keys(), values)) for values in itertools.product(*__inp.values()))
    else:
        space = (__inp)

    # init optimisation state
    best_params = dict()
    best_recsys = None
    best_recall = 0
    tr_data_csr = tr_data.csr_matrix  # for fast row access
    te_data_csr = te_data.csr_matrix  # for fast row access

    acc = 0
    while 1:
        acc += 1

        # sample hyperparameters dict
        if gridsearch:
            try:
                params = next(space)
            except StopIteration:
                break
        else:
            if acc >= n_iters:
                break
            params = {k: rand_member_nb(space[k]) for k in space.keys()}

        # update tied hyperparameters
        for k, k_tied in space_tied_constraints.items():
            params[k] = params[k_tied]

        # optimise
        recsys = model.clone(params)
        recsys.fit(tr_data, te_data)
        # estimate performance for each user in test data
        Q = []
        scores = np.zeros(tr_data.num_items, dtype="f8")
        for n in range(te_data.num_users):
            tr_cn_true = tr_data_csr.getrow(n).indices.astype("i8")
            te_cn_true = te_data_csr.getrow(n).indices.astype("i8")
            # get scores and down weight already chosen train items
            scores[:] = np.squeeze(recsys.score(n))
            scores[tr_cn_true] = -100000
            # calculate test items
            te_cn_pred = argtopk(scores, k=TOPK)
            Q.append(myrecall(te_cn_true, te_cn_pred))

        if (recall := np.mean(Q)) > best_recall:
            best_params = params
            best_recsys = recsys
            best_recall = recall
        del recsys, Q
        print(f"recall={recall:.4f}  "
              f"params={formatdict(params)}  |  best_recall={best_recall:.4f}  best_params={formatdict(best_params)}")

    return best_params, best_recsys, best_recall


# -----------------------------------------------------------------------------


_pop = cornac.models.MostPop(
    name="POP"
)
_pop_search_space = {"name": ["POP"]}
_pop_search_space_tied_constraints = {}


# -----------------------------------------------------------------------------


_wmf = cornac.models.WMF(
    name="WMF",
    k=HIDDEN_SIZE,
    lambda_u=0.01,
    lambda_v=0.01,
    a=1,                     # the confidence (c_nj) of collected ratings a.k.a. positive feedback
    b=0.01,                  # the confidence (c_nj) of unseen ratings
    learning_rate=1e-3,
    batch_size=96,           # how many random unique items to consider in one batch (num_users x batch_size)
    max_iter=1000,
    verbose=0,
)

_wmf_search_space = {
    "lambda_u"     : np.logspace(-2.0, 1.0, num=10, base=10),
    "b"            : np.logspace(-4.0, 0.0, num=10, base=10),
    "learning_rate": (1e-4, 1e-3, 1e-2),
}
_wmf_search_space_tied_constraints = {"lambda_v": "lambda_u"}  # tied parameter


# -----------------------------------------------------------------------------


_bpr = cornac.models.BPR(
    name="BPR",
    k=HIDDEN_SIZE,
    use_bias=True,
    lambda_reg=0.01,
    learning_rate=1e-3,
    max_iter=9999,          # should be linearly proportional to a nnz in data
    verbose=0,
    num_threads=1,
)

_bpr_search_space = {"lambda_reg": np.logspace(-4.0, 0.0, num=20, base=10), "learning_rate": (1e-4, 1e-3, 1e-2)}
_bpr_search_space_tied_constraints = {}


# -----------------------------------------------------------------------------


_vae = cornac.models.VAECF(
    name="VAE",
    k=HIDDEN_SIZE,
    autoencoder_structure=[HIDDEN_SIZE],
    act_fn="tanh",
    likelihood="mult",
    n_epochs=300,
    batch_size=1024,         # how many random unique users to consider in one batch (batch_size x num_items)
    learning_rate=1e-3,
    beta=1.0,
    verbose=0,
)

_vae_search_space = {"beta": np.linspace(0.5, 1.5, num=20), "learning_rate": (1e-4, 1e-3, 1e-2)}
_vae_search_space_tied_constraints = {}


# -----------------------------------------------------------------------------


_ease = cornac.models.EASE(
    name="EASE",
    lamb=1.0,
    posB=False,
    verbose=0,
)

_ease_search_space = {"lamb": np.logspace(2.0, 2.9, num=20, base=10)}
_ease_search_space_tied_constraints = {}


# -----------------------------------------------------------------------------


_lgcn = cornac.models.LightGCN(
    name="LightGCN",
    emb_size=HIDDEN_SIZE,
    num_layers=3,
    num_epochs=300,
    learning_rate=1e-3,
    batch_size=1024,
    early_stopping=None,
    lambda_reg=0.01,
    verbose=0,
)

_lgcn_search_space = {"lambda_reg": np.logspace(-4.0, 0.0, num=20, base=10), "learning_rate": (1e-4, 1e-3, 1e-2)}
_lgcn_search_space_tied_constraints = {}


# ----------------------------------- RECBOLE ---------------------------------


# recbole parameters
setup_args = {

    # environment
    "seed": 2025,
    "worker": 2,
    "reproducibility": True,
    "shuffle": True,
    "save_dataset": False,
    "save_dataloaders": False,
    "checkpoint_dir": "./log/",
    "state": "ERROR",
    "log_wandb": False,
    "log_tensorboard": False,

    # train
    "epochs": 300,
    "train_batch_size": 2048,
    "learner": "adam",
    "learning_rate": 0.001,
    "train_neg_sample_args": { "distribution": "uniform", "sample_num": 10, "dynamic": False },
    "eval_step": 300,
    "stopping_step": 10,
    "weight_decay": 0.001,

    # evaluation
    "eval_args": { "group_by": "user", "order": "RO", "split": {"RS": [7,3,0]}, "mode": "full" },  # train,validation,test
    "metrics": "Recall",
    "topk": [5,10,20],
    "valid_metric": "Recall@20",
    "eval_batch_size": 4096,
    "repeatable": False,
}


def make_and_save_atomic_data_from_y(y: np.ndarray, name: str) -> None:

    N = y.shape[0]
    J = y.shape[1]

    # init core atomic data
    a = np.zeros(shape=(np.sum(y > 0), 2), dtype=np.int32)

    t = 0
    for n in range(N):
        for j in range(J):
            # only implicit feedback
            if y[n,j] > 0: a[t,:] = [n,j]; t += 1

    a_tb = pl.DataFrame(data=a, schema={"user_id:token": pl.Int32,
                                        "item_id:token": pl.Int32})

    # create leaf directory if it does not exist
    os.makedirs(f"./out/recbole/{name}", mode=0o777, exist_ok=True)

    data_args = {
        "data_path": f"./out/recbole/",
        "dataset": f"{name}",
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "load_col": { "inter": ["user_id", "item_id"] },
    }

    # save user-item implicit feedback intereaction data to file
    a_tb.write_csv(f"./out/recbole/{name}/{name}.inter", separator="\t")
    # save data config to YAML file
    with open(f"./out/recbole/{name}/config.yaml", "w") as f: yaml.dump(data_args, f, default_flow_style=False)


def make_and_save_atomic_data(city: str) -> None:
    # load LBSN data
    y, _, _, _, _ = load_lbsn_artefacts(f"./out/{city}_lbsn.hdf5")
    make_and_save_atomic_data_from_y(y, city)


def SimpleX_trial(trial: optuna.trial.Trial) -> dict[str,any]:
    return {
        "embedding_size": HIDDEN_SIZE,
        "reg_weight": 0,
        "margin": trial.suggest_float("margin", -1, 1),
        "negative_weight": trial.suggest_int("negative_weight", 1, 100),
        "gamma": trial.suggest_float("gamma", 0, 1),
        "aggregator": trial.suggest_categorical("aggregator", ["mean", "user_attention", "self_attention"]),
        "history_len": trial.suggest_int("history_len", 5, 20),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "learning_rate": trial.suggest_categorical("learning_rate", (1e-4, 1e-3, 1e-2)),
    }


def DiffRec_trial(trial: optuna.trial.Trial) -> dict[str,any]:
    return {
        "noise_schedule": "linear",
        "noise_scale": trial.suggest_discrete_uniform("noise_scale", 1e-3, 1e-1, 1e-3),
        "noise_min": trial.suggest_discrete_uniform("noise_min", 1e-4, 1e-3, 1e-4),
        "noise_max": trial.suggest_discrete_uniform("noise_max", 1e-3, 1e-1, 1e-3),
        "reweight": True,
        "steps": trial.suggest_int("steps", 2, 10),
        "history_num_per_term": trial.suggest_int("history_num_per_term", 1, 10),
        "dims_dnn": ast.literal_eval(trial.suggest_categorical("dims_dnn", ["[32]", "[64]", "[128]", "[256]"])),
        "embedding_size": trial.suggest_categorical("embedding_size", [4, 8, 16]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "learning_rate": trial.suggest_categorical("learning_rate", (1e-4, 1e-3, 1e-2)),
    }


def get_model(name: str):
    """Map model names to model classes."""
    match name:
        case "SimpleX":
            return recbole.model.general_recommender.SimpleX
        case "DiffRec":
            return recbole.model.general_recommender.DiffRec


def rb_fit(config, update_args):

    # get model and new config
    config = recbole.config.Config(config["model"], config["dataset"],
                                   config_dict=copy.deepcopy(config.external_config_dict) | update_args)

    dataset = recbole.data.create_dataset(config)
    tr_data, va_data, te_data = recbole.data.data_preparation(config, dataset)

    net = get_model(config["model"])(config, tr_data.dataset)
    trainer = recbole.trainer.Trainer(config, net)

    # fit model
    best_va_score, best_va_result = trainer.fit(tr_data,
                                                va_data,
                                                verbose=False,
                                                saved=False,
                                                show_progress=False,
                                                )
    return net, (tr_data, va_data, te_data), (best_va_score, best_va_result)


def rb_objective(trial: optuna.trial.Trial, config: recbole.config.configurator.Config) -> float:
    """Objective function for Optuna to optimize. A trial is a single call to this objective function."""

    match config["model"]:
        case "SimpleX":
            trial_sampled_args = SimpleX_trial(trial)
        case "DiffRec":
            trial_sampled_args = DiffRec_trial(trial)
        case _:
            raise ValueError(f"Wrong model name given: {config['model']}")

    try:
        net, _, (best_va_score, best_va_result) = rb_fit(config, update_args=trial_sampled_args)
        print(f"Trial {trial.number} (valid) {config['valid_metric']}={best_va_score:.4f} params: {trial.params}")
        return best_va_score

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return 0.0
