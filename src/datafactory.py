import os
import h5py
import numpy as np


def dump_lbsn_artefacts(file: str, y: np.ndarray, t: np.ndarray, f: np.ndarray, fnames: np.ndarray, envstruct: dict, _str: str = "S8"):

    assert len(f) == len(fnames), "Shape mismatch"

    # create leaf directory if it does not exist
    os.makedirs(os.path.dirname(file), mode=0o777, exist_ok=True)

    with h5py.File(file, "w") as o:
        # create environment storage for meta-information data
        o.create_group("/environment")
        o["/environment"].attrs["city_centre"] = envstruct["city_centre"]
        o["/environment"].attrs["city_radius"] = envstruct["city_radius"]
        o["/environment"].create_dataset("location", dtype="f8", data=envstruct["location"])
        o["/environment"].create_dataset("capacity", dtype="i8", data=envstruct["capacity"])
        o["/environment"].create_dataset("time_required", dtype="i8", data=envstruct["time_required"])
        # create users and items storage
        o["/"].create_dataset("y", dtype="i8", data=y)
        o["/"].create_dataset("t", dtype="i8", data=t)  # Unix epoch time
        o["/"].create_dataset("f", dtype="f8", data=f)
        o["/"].create_dataset("fnames", dtype=_str, data=fnames)
    print(f"Dumped successfully lbsn artefacts ==> {os.path.basename(file)}")


def load_lbsn_artefacts(file: str, _str: str = "U"):

    with h5py.File(file, "r") as o:
        # read meta-information from environment storage
        g = o["/environment"]
        envstruct = {
            "city_centre": g.attrs["city_centre"],
            "city_radius": g.attrs["city_radius"],
            "location": g["location"][...].astype("f8"),
            "capacity": g["capacity"][...].astype("i8"),
            "time_required": g["time_required"][...].astype("i8"),
        }
        # read users and items storage
        y = o["/y"][...].astype("i8")
        t = o["/t"][...].astype("i8")
        f = o["/f"][...].astype("f8")
        fnames = o["/fnames"].asstr()[...].astype(_str)
    print(f"Loaded successfully lbsn artefacts <== {os.path.basename(file)}")

    return y, t, f, fnames, envstruct


def dump_util_artefacts(file: str, true_data_map: dict[str,np.ndarray], pred_data_map: dict[str,dict[str,np.ndarray]]):

    # create leaf directory if it does not exist
    os.makedirs(os.path.dirname(file), mode=0o777, exist_ok=True)

    with h5py.File(file, "w") as o:
        # create storage for input data with ground truth (true) and predicted preferences (pred)
        o.create_group("/true")
        o.create_group("/pred")
        # populate true storage
        for key, arr in true_data_map.items():
            o["/true"].create_dataset(
                name=key,
                data=arr,
            )
        # populate pred storage
        for key, arr in pred_data_map.items():
            print(f"Processing data storage with key [{key:<7}]...", end=" ")
            o["/pred"].create_group(key)
            for t_key, t_arr in arr.items():
                o[f"/pred/{key}"].create_dataset(
                    name=t_key,
                    data=t_arr,
                )
            print(f"DONE.")
    print(f"Dumped successfully reco artefacts ==> {os.path.basename(file)}")


def load_util_artefacts(file: str):
    """
    hdf:
     |__ true:
     |     |__ y  : y
     |     |__ wmf: u
     |     |__ bpr: u
     |     |__ xxx: u
     |
     |__ pred:
           |__ y  : {"t=2": y, "t=3": y, "t=4": y}
           |__ wmf: {"t=2": u, "t=3": u, "t=4": u}
           |__ bpr: {"t=2": u, "t=3": u, "t=4": u}
           |__ xxx: {"t=2": u, "t=3": u, "t=4": u}
    """

    true_data_map: dict[str,np.ndarray] = {}
    pred_data_map: dict[str,dict[str,np.ndarray]] = {}

    with h5py.File(file, "r") as o:
        # read true storage
        for key, arr in o["/true"].items():
            true_data_map[key] = arr[...]
        # read pred storage
        for key, arr in o["/pred"].items():
            print(f"Processing data storage with key [{key:<7}]...", end=" ")
            pred_data_map[key] = {t_key: t_arr[...]
                                  for (t_key, t_arr) in arr.items()}
            print(f"DONE.")
    print(f"Loaded successfully reco artefacts <== {os.path.basename(file)}")
    return true_data_map, pred_data_map


def dump_aset_artefacts(file: str, y: np.ndarray, A_map: dict[str,dict[str,np.ndarray]]):
    # create leaf directory if it does not exist
    os.makedirs(os.path.dirname(file), mode=0o777, exist_ok=True)
    with h5py.File(file, "w") as o:
        o["/"].create_dataset("y", dtype="i8", data=y)
        for key, arr in A_map.items():
            o["/"].create_group(key)
            for t_key, t_arr in arr.items():
                o[f"/{key}"].create_dataset(
                    name=t_key,
                    data=t_arr,
                )
    print(f"Dumped successfully aset artefacts ==> {os.path.basename(file)}")


def load_aset_artefacts(file: str):
    """
    hdf:
     |__ y
     |__ wmf: {"A": arr, "beta": arr, "gamma": arr, "u_thr": arr}
     |__ bpr: {"A": arr, "beta": arr, "gamma": arr, "u_thr": arr}
     |__ xxx: {"A": arr, "beta": arr, "gamma": arr, "u_thr": arr}
    """

    A_map = {}
    with h5py.File(file, "r") as o:
        y = o["/y"][...].astype("i8")
        for key, arr in o["/"].items():
            if key != "y":
                print(f"Processing data storage with key [{key:<7}]...", end=" ")
                A_map[key] = {t_key: t_arr[...]
                              for (t_key, t_arr) in arr.items()}
                print(f"DONE.")
    print(f"Loaded successfully aset artefacts <== {os.path.basename(file)}")
    return y, A_map
