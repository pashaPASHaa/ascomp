import argparse
import subprocess
import sys
import numpy as np

from datafactory import load_util_artefacts, dump_util_artefacts


def build_cmd(script, city, seed, suffix):
    """Builds a shell command string that runs the given script and tees its combined stdout+stderr."""
    lbsn_file = f"./out/{city}_lbsn.hdf5"
    util_file = f"./out/{city}_{seed}_{suffix}_util.hdf5"
    dlog_file = f"./log/{city}_{seed}_{suffix}_run1_training_dump.log"
    n_threads = 4
    return (
        f"OMP_NUM_THREADS={n_threads} "
        f"MKL_NUM_THREADS={n_threads} "
        f"OPENBLAS_NUM_THREADS={n_threads} "
        f"TF_NUM_INTEROP_THREADS={n_threads} "
        f"TF_NUM_INTRAOP_THREADS={n_threads} "
        f"python3 -u {script} "
        f"--lbsn_artefacts_file {lbsn_file} "
        f"--util_artefacts_file {util_file} "
        f"--seed {seed} "
        f"2>&1 | tee {dlog_file}",
        util_file,
    )


def main():
    # Script accepts one or more integers:
    # python3 -u src/run1.py --city Rome --seed_list 2026 2027
    # python3 -u src/run1.py --city Rome --seed_list 2026
    parser = argparse.ArgumentParser(description="Run both src/run1_co.py and run1_rb.py in parallel")
    parser.add_argument("--city", type=str, required=True, help="city")
    parser.add_argument("--seed_list", type=int, nargs="+", required=True, help="List of seed values (e.g. --seed_list 1 2 3)")
    args = parser.parse_args()

    for seed in args.seed_list:
        print(f"Processing city={args.city} seed={seed}")

        # prepare the two commands
        cmd1, util_file1 = build_cmd(script="src/run1_co.py", city=args.city, seed=seed, suffix="co")
        cmd2, util_file2 = build_cmd(script="src/run1_rb.py", city=args.city, seed=seed, suffix="rb")

        # launch both in parallel (shell=True so we can use the pipe+tee)
        print(f"Launching:\n  {cmd1}\n  {cmd2}\n")
        p1 = subprocess.Popen(cmd1, shell=True, executable="/bin/bash")
        p2 = subprocess.Popen(cmd2, shell=True, executable="/bin/bash")

        # wait for both to complete
        ret1 = p1.wait()
        ret2 = p2.wait()

        if ret1 != 0 or ret2 != 0:
            print(f"One or both processes failed (codes: {ret1}, {ret2})", file=sys.stderr)
            sys.exit(max(ret1, ret2))

        # merge
        true_data_map1, pred_data_map1 = load_util_artefacts(util_file1)
        true_data_map2, pred_data_map2 = load_util_artefacts(util_file2)

        assert np.all(
            true_data_map1["y"] == true_data_map2["y"]), "Mismatch in true labels (y) between co and rb results"

        true_data_map = true_data_map1 | true_data_map2
        pred_data_map = pred_data_map1 | pred_data_map2

        # and dump merged results to disk
        dump_util_artefacts(
            f"./out/{args.city}_{seed}_util.hdf5",
            true_data_map,
            pred_data_map,
        )

        print(f"Both runs (co:cornac & rb:recbole) completed successfully, results merged")
    sys.exit(0)


if __name__ == "__main__":
    main()
