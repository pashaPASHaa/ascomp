#!/bin/bash

# check if a city argument is provided
if [ -z "$1" ]; then
  echo "Error: City argument is missing."
  exit 1
fi


city="$1"


# hardcoded list of seeds
seeds=(2025 2026 2027 2028 2029)


for seed in "${seeds[@]}"; do
  echo "Launching run2.py for city=${city}, seed=${seed}"
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TF_NUM_INTEROP_THREADS=4 TF_NUM_INTRAOP_THREADS=4 \
  python3 -u src/run2.py \
          --util_artefacts_file out/${city}_${seed}_util.hdf5 \
          --aset_artefacts_file out/${city}_${seed}_aset.hdf5 \
	  --topk 10 \
	  --seed ${seed} \
          2>&1 | tee log/${city}_${seed}_run2_training.log &
done
wait
echo "All processes run2.py completed."
