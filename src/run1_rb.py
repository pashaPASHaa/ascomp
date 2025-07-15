import argparse
import functools
import os
import warnings

import numpy as np
import optuna
import recbole
import recbole.config
import recbole.data
import recbole.data.interaction
import recbole.model.general_recommender
import recbole.trainer
import torch

from collaborative_filtering import setup_args, make_and_save_atomic_data, make_and_save_atomic_data_from_y, rb_fit, rb_objective
from datafactory import load_lbsn_artefacts, dump_util_artefacts
from utils import argnonz, argtopk, myrecall, seedutils, rand_choice_arr_nb

warnings.filterwarnings("ignore", category=FutureWarning)  # pandas warning


# how many items recommender engine knows about?
gathered_k_choice_arr = (2, 3, 4, 5, 50)


def get_U(net, net_tr_data, N, J):  # --> estimate scores for each user-item pair

    assert (
        1
        and net.n_users == N+1  # (user with index 0 is a [pad] token)
        and net.n_items <= J+1  # (item with index 0 is a [pad] token)
    ), "Wrong input!"

    # get scores for known items and known users
    known_users = recbole.data.interaction.Interaction({
        "user_id": net_tr_data.dataset.token2id("user_id", [str(token) for token in range(0, net.n_users-1)]),
    })

    net.eval()
    with torch.no_grad():
        u = net.full_sort_predict(known_users).cpu().numpy()[:,1:]  # need skip [pad] item index (0)
    # normalise utility vector [0,1], that does not affect ranking metrics!
    u -= u.min(1, keepdims=True)
    u /= u.max(1, keepdims=True)

    # map model scores to the full matrix
    # let unknown user-item interactions have zero score in the final utility matrix
    u_full = np.full(shape=(N, J), fill_value=0, dtype=np.float32)

    # model internal IDs are 1-index but original item IDs in the .inter file are 0-index integers
    orig_token_list = [int(net_tr_data.dataset.id2token("item_id", internal_id)) for internal_id in range(1, net.n_items)]
    # put scores into the correct column
    u_full[:,orig_token_list] = u[:,:]

    return u_full


if __name__ == "__main__":

    # -------------------------------------------------------------------------

    # python3 run1_rb.py --help
    parser = argparse.ArgumentParser(description="Synthetic preference estimation protocol")
    parser.add_argument("--lbsn_artefacts_file", type=str, required=True, help="file with LBSN itineraries")
    parser.add_argument("--util_artefacts_file", type=str, required=True, help="file output")
    parser.add_argument("--seed", type=int, required=True, help="seed")
    args = parser.parse_args()

    # -------------------------------------------------------------------------

    city = os.path.basename(args.lbsn_artefacts_file).split("_")[0]  # Rome, Florence, London, ...

    # reproducibility
    np.random.seed(args.seed)
    seedutils(args.seed)
    setup_args["seed"] = args.seed
    # load file
    y, _, _, _, _ = load_lbsn_artefacts(args.lbsn_artefacts_file)  # y[n,j] = {0,1}
    # data configuration: n_users, n_items
    N, J = y.shape
    # distribution for t-feedback sampling
    distribution_pop = y.sum(axis=0) / y.sum()

    print(f"Detected script configuration for synthetic preference estimation protocol:\n"
          f"N={N} [users] and J={J} [items] where the "
          f"most popular item p={distribution_pop[0]:.4f} and the most niche item p={distribution_pop[-1]:.4f}")

    # -------------------------------------------------------------------------

    # containers with artefacts
    CF_true_data_map: dict[str,np.ndarray] = {"y": y}

    make_and_save_atomic_data(city)
    CF_map = {"SimpleX": {}, "DiffRec": {}}

    # -------------------------------------------------------------------------

    print(f"\nHYPERPARAMETERS SEARCH\n")

    # find the best hyperparameters
    for key in CF_map:

        config = recbole.config.Config(
            model=key,
            dataset=city,
            config_file_list=[f"./out/recbole/{city}/config.yaml"],
            config_dict=setup_args,
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=args.seed),  # tree-structured Parzen estimator
        )

        print(f"\nStarting hyperparameter optimization")
        study.optimize(
            func=functools.partial(rb_objective, config=config),
            n_trials=150,
            show_progress_bar=False,
            gc_after_trial=True,
        )

        CF_map[key]["best_params"] = study.best_trial.params
        print(f"\n")
        print(f"Optimization for {key} DONE. {config['valid_metric']} on validation split={study.best_trial.value:.4f}. "
              f"Total trials={len(study.trials)} and best trial={study.best_trial.number}")
        print(f"Best hyperparameters:")
        for p, v in study.best_trial.params.items():
            print(f"\t{p}:{v}")

        # measure quality of fit & compute preference vectors for each user
        net, (tr_data, va_data, te_data), _ = rb_fit(config, update_args=study.best_trial.params)
        tr_coo_data = (
            tr_data.dataset.inter_matrix().toarray()
            [1:,1:][:,tr_data.dataset.token2id("item_id", [str(token) for token in range(0, net.n_items-1)] ) - 1]
            .copy(order="C")
        )
        va_coo_data = (
            va_data.dataset.inter_matrix().toarray()
            [1:,1:][:,va_data.dataset.token2id("item_id", [str(token) for token in range(0, net.n_items-1)] ) - 1]
            .copy(order="C")
        )
        u_pred = get_U(net, tr_data, N, J).astype(np.float64)  # oracle preferences
        del net
        CF_true_data_map[key] = np.copy(u_pred)

        print("Debug information:")
        print(tr_coo_data.sum(0))
        print(va_coo_data.sum(0))
        with np.printoptions(precision=3, suppress=True):
            print(u_pred.mean(0))

        Qat10 = []
        Qat20 = []
        for n in range(N):
            tr_cn = argnonz(tr_coo_data[n])
            va_cn = argnonz(va_coo_data[n])
            # mask train items
            u_pred[n,tr_cn] = -100000
            # eval recall only on test items
            Qat10.append(myrecall(va_cn, argtopk(u_pred[n,:], k=10)))
            Qat20.append(myrecall(va_cn, argtopk(u_pred[n,:], k=20)))
        print(f"--------------------\n"
              f"recall@10={np.mean(Qat10):.4f} +/- {np.std(Qat10):.4f}\n"
              f"recall@20={np.mean(Qat20):.4f} +/- {np.std(Qat20):.4f}\n",
              flush=True)

    # -------------------------------------------------------------------------

    # containers with artefacts
    CF_pred_data_map: dict[str,dict[str,np.ndarray]] = {"y": {}, **{key: {} for key in CF_map}}

    # -------------------------------------------------------------------------

    print(f"\nESTIMATING PARTIAL UTILITY (BASED ON K GATHERED CHOICES)\n")

    # model limited knowledge about user with partially revealed feedback
    for k in gathered_k_choice_arr:

        y_pred = np.zeros(y.shape, dtype="i8")  # --> (N,J) imitate partial y
        u_pred = np.zeros(y.shape, dtype="f8")  # --> (N,J) imitate partial utility

        for n in range(N):
            hot_arr = argnonz(y[n,:])  # index of true choices
            hot_len = hot_arr.size
            if hot_len > 0:
                # assume that preference extraction is related to popularity of items
                # popularity bias facilitates the collection of preferences for major (mainstream) products
                y_pred[
                    n,
                    hot_arr[rand_choice_arr_nb(
                        p=distribution_pop[hot_arr] / distribution_pop[hot_arr].sum(), size=min(k, hot_len))]
                ] = 1
        CF_pred_data_map["y"][str(k)] = np.copy(y_pred)

        # preference learning mechanism based on collaborative filtering (CF) techniques
        # only implicit feedback is available in collected data =>
        # CF algorithms should work with implicit {0,1} signal
        make_and_save_atomic_data_from_y(y_pred, f"{city}{k}")

        for key in CF_map:
            # reload config for a limited knowledge version of orig dataset
            config = recbole.config.Config(
                model=key,
                dataset=f"{city}{k}",
                config_file_list=[f"./out/recbole/{city}{k}/config.yaml"],
                config_dict=setup_args,
            )
            # fit
            net, (data, _, _), _ = rb_fit(
                config, update_args=CF_map[key]["best_params"] |
                                    {"eval_args": {"group_by": "user", "order": "RO", "split": {"RS": [1,0,0]}, "mode": "full"}}
            )
            # score user preferences for each (n,j)-pair
            u_pred[:,:] = get_U(net, data, N, J)
            CF_pred_data_map[key][str(k)] = np.copy(u_pred)
            del net
        print(f"Fit {k}-items DONE.")

    # -------------------------------------------------------------------------

    # dump trained artefacts in file for later use
    dump_util_artefacts(args.util_artefacts_file, true_data_map=CF_true_data_map, pred_data_map=CF_pred_data_map)

    # -------------------------------------------------------------------------

    print(f"DONE!")
