import argparse
import os
import subprocess
import time


MAX_CONCURRENT = 32


def make_runx_instruction(this_path, city, seed, key, recommender_system_key):
    runner = (
        f"python3 {this_path}/runx.py "
        f"--lbsn_artefacts_file {this_path}/../out/{city}_lbsn.hdf5 "
        f"--util_artefacts_file {this_path}/../out/{city}_{seed}_util.hdf5 "
        f"--aset_artefacts_file {this_path}/../out/{city}_{seed}_aset.hdf5 "
        f"--dump_file {this_path}/../out/experiments/runx_{city}_{seed}_{key}_{recommender_system_key}.pk "
        f"--key {key} "
        f"--recommender_system_key {recommender_system_key} "
        f"--seed {seed} "
        f"2>&1 | tee {this_path}/../log/runx_{city}_{seed}_{key}_{recommender_system_key}.log"
    )
    return runner


if __name__ == "__main__":

    # Script accepts one or more integers:
    # python3 -u src/runx_batch.py --city Rome --seed_list 2026 2027
    # python3 -u src/runx_batch.py --city Rome --seed_list 2026
    parser = argparse.ArgumentParser(description="Batch Experiment")
    parser.add_argument("--city", type=str, required=True, help="City of interest: Rome, Florence, Istanbul, etc.")
    parser.add_argument("--seed_list", type=int, nargs="+", required=True, help="List of seed values (e.g. --seed_list 1 2 3)")
    args = parser.parse_args()

    this_path = os.path.dirname(__file__)
    keys = ["pop", "wmf", "bpr", "vae", "lgcn", "ease", "SimpleX", "DiffRec"]  # design of experiment

    active = []
    for seed in args.seed_list:
        for key in keys:
            for recommender_system_key in keys:
                print(f"Starting experiment in city={args.city} seed={seed} with configuration parameters "
                      f"key={key:>4} "
                      f"recommender_system_key={recommender_system_key:>4}")

                cmdx = make_runx_instruction(this_path=this_path,
                                             city=args.city, seed=seed, key=key, recommender_system_key=recommender_system_key)

                proc = subprocess.Popen(cmdx,
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL,
                                        shell=True,
                                        executable="/bin/bash")
                active.append(proc)

                # if we have hit the limit, wait for *one* to finish before launching more
                while len(active) >= MAX_CONCURRENT:
                    for proc in active:
                        if proc.poll() is not None:  # process has exited
                            active.remove(proc)
                            break
                    else:
                        time.sleep(1)  # none finished yet -> wait and retry

    # finally, wait for the rest to finish
    for proc in active:
        proc.wait()

    print(f"DONE!")
