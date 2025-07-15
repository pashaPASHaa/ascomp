import argparse
import collections
import numpy as np

from collaborative_filtering import _pop, _pop_search_space, _pop_search_space_tied_constraints
from collaborative_filtering import _vae, _vae_search_space, _vae_search_space_tied_constraints
from collaborative_filtering import _wmf, _wmf_search_space, _wmf_search_space_tied_constraints
from collaborative_filtering import _bpr, _bpr_search_space, _bpr_search_space_tied_constraints
from collaborative_filtering import _ease, _ease_search_space, _ease_search_space_tied_constraints
from collaborative_filtering import _lgcn, _lgcn_search_space, _lgcn_search_space_tied_constraints
from collaborative_filtering import formatdict
from collaborative_filtering import hsearch, prepare_positive_feedback_data, prepare_and_split_positive_feedback_data
from datafactory import load_lbsn_artefacts, dump_util_artefacts
from utils import argnonz, argtopk, myrecall, rand_choice_arr_nb, seedutils


# all implicit collaborative filtering (CF) algorithms to consider
CF_map = {
    "lgcn": _lgcn,
    "ease": _ease,
    "pop": _pop,
    "vae": _vae,
    "wmf": _wmf,
    "bpr": _bpr,
}
CF_search_space_map = {
    "lgcn": (_lgcn_search_space, _lgcn_search_space_tied_constraints),
    "ease": (_ease_search_space, _ease_search_space_tied_constraints),
    "pop": (_pop_search_space, _pop_search_space_tied_constraints),
    "vae": (_vae_search_space, _vae_search_space_tied_constraints),
    "wmf": (_wmf_search_space, _wmf_search_space_tied_constraints),
    "bpr": (_bpr_search_space, _bpr_search_space_tied_constraints),
}


# how many items recommender engine knows about?
gathered_k_choice_arr = (2, 3, 4, 5, 50)


def get_U(model, data):  # --> estimate scores for users in `data`

    assert (model.num_users == data.num_users) and (model.uid_map == data.uid_map), "Shape or Map mismatch"
    assert (model.num_items == data.num_items) and (model.iid_map == data.iid_map), "Shape or Map mismatch"

    N = data.num_users
    J = data.num_items
    u = np.empty((N, J), dtype="f8")

    for n in data.uid_map:
        # score items
        u[n,:] = model.score(n)
        # normalise utility vector [0,1], that does not affect ranking metrics!
        u[n,:] -= u[n,:].min()
        u[n,:] /= u[n,:].max()
    return u


if __name__ == "__main__":

    # -------------------------------------------------------------------------

    # python3 run1_co.py --help
    parser = argparse.ArgumentParser(description="Synthetic preference estimation protocol")
    parser.add_argument("--lbsn_artefacts_file", type=str, required=True, help="file with LBSN itineraries")
    parser.add_argument("--util_artefacts_file", type=str, required=True, help="file output")
    parser.add_argument("--seed", type=int, required=True, help="seed")
    args = parser.parse_args()

    # -------------------------------------------------------------------------

    # reproducibility
    np.random.seed(args.seed)
    seedutils(args.seed)
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

    # global map
    uid_map = collections.OrderedDict([(n, n) for n in range(N)])  # orig user id (any) --> internal user idx (int)
    iid_map = collections.OrderedDict([(j, j) for j in range(J)])  # orig item id (any) --> internal item idx (int)

    # containers with artefacts
    CF_true_data_map: dict[str,np.ndarray] = {"y": y}

    # make data for preference estimation
    true_data = prepare_positive_feedback_data(y, uid_map, iid_map, seed=args.seed)

    # -------------------------------------------------------------------------

    print(f"\nHYPERPARAMETERS SEARCH\n")

    # make data split
    tr_data, te_data = prepare_and_split_positive_feedback_data(y, uid_map, iid_map, seed=args.seed, te_frac=0.33)

    # find the best hyperparameters
    CF_best_params_map: dict[str,dict] = {}
    for key in CF_map:
        CF_best_params_map[key], best_recsys, best_recall = hsearch(
            CF_map[key],
            CF_search_space_map[key][0],
            CF_search_space_map[key][1],
            tr_data,
            te_data,
            gridsearch=True)
        print(f"Search for {key} DONE. recall={best_recall:.4f} params={formatdict(CF_best_params_map[key])}")
        # compute preference vectors for each user
        CF_true_data_map[key] = get_U(best_recsys, true_data)  # <-- will be used for oracle choice simulation
        # measure quality of fit
        Qat10 = []
        Qat20 = []
        te_pred = get_U(best_recsys, te_data)
        for n in range(N):
            tr_cn = np.array(tr_data.user_data[n][0], dtype="i8")
            te_cn = np.array(te_data.user_data[n][0], dtype="i8")
            # mask train items
            te_pred[n,tr_cn] = -100000
            # eval recall just at test items
            Qat10.append(myrecall(te_cn, argtopk(te_pred[n,:], k=10)))
            Qat20.append(myrecall(te_cn, argtopk(te_pred[n,:], k=20)))
        print(f"--------------------\n"
              f"recall@10={np.mean(Qat10):.4f} +/- {np.std(Qat10):.4f}\n"
              f"recall@20={np.mean(Qat20):.4f} +/- {np.std(Qat20):.4f}\n")
    del tr_data, te_data

    # -------------------------------------------------------------------------

    print(f"\nESTIMATING PARTIAL UTILITY (BASED ON K GATHERED CHOICES)\n")

    # containers with artefacts
    CF_pred_data_map: dict[str,dict[str,np.ndarray]] = {"y": {}, **{key: {} for key in CF_map}}

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
        pred_data = prepare_positive_feedback_data(y_pred, uid_map, iid_map, seed=args.seed)

        for key in CF_map:
            # score user preferences for each (n,j)-pair
            recsys = CF_map[key].clone(CF_best_params_map[key])
            recsys.fit(pred_data)
            u_pred[:,:] = get_U(recsys, pred_data)
            CF_pred_data_map[key][str(k)] = np.copy(u_pred)
        del pred_data
        print(f"Fit {k}-items DONE.")

    # -------------------------------------------------------------------------

    # dump trained artefacts in file for later use
    dump_util_artefacts(args.util_artefacts_file, true_data_map=CF_true_data_map, pred_data_map=CF_pred_data_map)

    # -------------------------------------------------------------------------

    print(f"DONE!")
