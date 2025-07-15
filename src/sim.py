import time
import numpy as np

from datafactory import load_util_artefacts, load_aset_artefacts
from recommender import do_planning, recommend_with_void_search_decoder
from ubm import make_boltzman_choices
from utils import argnonz


def simulate_1_day(
        lbsn_file: str,
        util_file: str,
        aset_file: str,
        args: dict[str,any],
):
    # -------------------------------------------------------------------------

    # parameters that define:
    # - ground truth preference extraction mechanism
    # - algorithm to be used as a recommender engine
    key = args["key"]
    recommender_system_key, recommender_system_t_key = args["recommender_system_key"]

    # parameters that may affect the simulation outcome
    lamb = args["lamb"]  # tradeoff (1-lamb) * user_u + lamb * sust_u
    pack_size = args["pack_size"]
    salience_boost = args["salience_boost"]
    seed = args["seed"]
    disp_collector_step = args["disp_collector_step"]  # how often display info
    croi = args["croi"]

    # -------------------------------------------------------------------------

    # load dataset with user preferences
    true_data_map, pred_data_map = load_util_artefacts(util_file)

    # load dataset with user awareness set
    # hdf:
    #  |__ y
    #  |__ wmf: {"A": arr, "beta": arr, "gamma": arr, "u_thr": arr}
    #  |__ bpr: {"A": arr, "beta": arr, "gamma": arr, "u_thr": arr}
    #  |__ any: {"A": arr, "beta": arr, "gamma": arr, "u_thr": arr}
    # """

    y_true, A_map = load_aset_artefacts(aset_file)

    assert np.all(true_data_map["y"] == y_true), f"Data mismatch between {util_file} and {aset_file} file"

    N, J = y_true.shape
    true_user_preferences = true_data_map[key]
    pred_user_preferences = pred_data_map[recommender_system_key][recommender_system_t_key]
    A, beta, gamma, u_thr = A_map[key]["A"], A_map[key]["beta"], A_map[key]["gamma"], A_map[key]["u_thr"]
    del true_data_map

    # -------------------------------------------------------------------------

    # sustainable promotion utility
    advt_preferences = do_planning(J, croi)

    # -------------------------------------------------------------------------

    # init structure
    log = {
        "caus_user": np.zeros(N, dtype="f4"),
        "caus_harm": np.zeros(N, dtype="i4"),
        "nors_user": np.zeros(N, dtype="f4"),
        "nors_harm": np.zeros(N, dtype="i4"),
        "data_user": np.zeros(N, dtype="f4"),
        "data_harm": np.zeros(N, dtype="i4"),
        "n": np.zeros(N, dtype="i4"),
        "caus_choice_arr": np.zeros(J, dtype="i4"), "caus_exposure_arr": np.zeros(J, dtype="i4"),
        "nors_choice_arr": np.zeros(J, dtype="i4"),
        "data_choice_arr": np.zeros(J, dtype="i4"),
    }
    set_croi = set(croi)  # set of croi items, for fast set lookup & operations
    np.random.seed(seed)  # fix seed

    # -------------------------------------------------------------------------

    for n in range(N):

        tic = time.time()

        # ---------------------------------------------------------------------
        # RECOMMENDER SYSTEM FORESEES WHAT PACK TO RECOMMEND (AS A PROPOSITION)

        caus_pack = recommend_with_void_search_decoder(pred_user_preferences[n,:], advt_preferences, pack_size, lamb)
        nors_pack = np.array([], dtype="i8")
        # record recommendations
        log["caus_exposure_arr"][caus_pack] += 1

        # ---------------------------------------------------------------------
        # USER REACTS TO PROPOSED RECOMMENDATION PACK AND FOLLOWS THE ITINERARY

        anset, need_size = argnonz(A[n,:]), y_true[n,:].sum()

        boost = np.where(np.isin(np.arange(J), caus_pack), salience_boost, 0.0)
        accepted_caus_pack = make_boltzman_choices(true_user_preferences[n,:] + boost, caus_pack, anset,
                                                   need_size, beta[n], gamma[n], u_thr[n])

        boost = 0
        accepted_nors_pack = make_boltzman_choices(true_user_preferences[n,:] + boost, nors_pack, anset,
                                                   need_size, beta[n], gamma[n], u_thr[n])

        accepted_data_pack = argnonz(y_true[n,:])

        # -----------------------------

        # made choices increment
        log["caus_choice_arr"][accepted_caus_pack] += 1
        log["nors_choice_arr"][accepted_nors_pack] += 1
        log["data_choice_arr"][accepted_data_pack] += 1

        # -----------------------------

        caus_user = np.sum(true_user_preferences[n,accepted_caus_pack])
        nors_user = np.sum(true_user_preferences[n,accepted_nors_pack])
        data_user = np.sum(true_user_preferences[n,accepted_data_pack])
        # record utilities
        log["caus_user"][n] = caus_user
        log["nors_user"][n] = nors_user
        log["data_user"][n] = data_user
        log["caus_harm"][n] = len(set(accepted_caus_pack) & set_croi)
        log["nors_harm"][n] = len(set(accepted_nors_pack) & set_croi)
        log["data_harm"][n] = len(set(accepted_data_pack) & set_croi)

        # record user index
        log["n"][n] = n

        # -----------------------------

        toc = time.time()

        if (n % disp_collector_step) == 0:
            print(
                f"{n:>04d} | {toc-tic:>6.4f} sec  k={need_size:>2d}"
                f"  |tn|  "
                f"util= "
                f"{caus_user:>5.2f}  "
                f"{nors_user:>5.2f}  |  "
                f"harm= "
                f"{np.sum(log['caus_harm']):>6.0f}  "
                f"{np.sum(log['nors_harm']):>6.0f}  "
            )

    return {key: np.asarray(val) for key, val in log.items()}


if __name__ == "__main__":

    import os

    # -------------------------------->

    path = os.path.dirname(os.path.realpath(__file__))
    lbsn_file = os.path.join(path, f"../out/Pisa_lbsn.hdf5")
    util_file = os.path.join(path, f"../out/Pisa_2026_util.hdf5")
    aset_file = os.path.join(path, f"../out/Pisa_2026_aset.hdf5")

    # -------------------------------->

    croi = np.arange(25)  # first 25 POIs are considered as unsustainable

    # -------------------------------->

    simulate_1_day(
        lbsn_file,
        util_file,
        aset_file,
        args={
            "key": "bpr",
            "recommender_system_key": ("wmf", "4"),
            "lamb": 0.4,
            "pack_size": 8,
            "salience_boost": 0.01,
            "seed": 1,
            "disp_collector_step": 1,
            "croi": croi,
        }
    )
