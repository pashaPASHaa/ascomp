### Positive-Sum Impact of Multistakeholder Recommendations for Urban Tourism
This repository contains source code for replicating experiments for 
Applied Soft Computing (special issue, Recommender Systems: Methodology Update) Journal.

##### IN CASE YOU PLAN TO RUN EXPERIMENTS, EXECUTE THE FOLLOWING STEPS

0. (a). Download this repository and original data:
- <https://github.com/igobrilhante/TripBuilder> (full)
- <https://sites.google.com/site/yangdingqi/home/foursquare-dataset> (global-scale check-in dataset)

0. (b). Prepare directory structure and unpack data:
```
(ascomp)
|-- data
|   |-- Foursquare33M
|   |   |-- dataset_TIST2015_Checkins.txt
|   |   |-- dataset_TIST2015_Cities.txt
|   |   |-- dataset_TIST2015_POIs.txt
|   |   `-- dataset_TIST2015_readme.txt
|   `-- tripbuilder-dataset-dist
|       |-- assets
|       |   |-- ...
|       |   `-- license.txt
|       |-- florence
|       |   |-- florence-photos.txt
|       |   |-- florence-pois-clusters.txt
|       |   |-- florence-pois.txt
|       |   |-- florence-trajectories.txt
|       |   `-- license.txt
|       |-- index.html
|       |-- license.txt
|       |-- pisa
|       |   |-- license.txt
|       |   |-- pisa-photos.txt
|       |   |-- pisa-pois-clusters.txt
|       |   |-- pisa-pois.txt
|       |   `-- pisa-trajectories.txt
|       `-- rome
|           |-- license.txt
|           |-- rome-photos.txt
|           |-- rome-pois-clusters.txt
|           |-- rome-pois.txt
|           `-- rome-trajectories.txt
|-- log
|-- notebooks
    `-- plot.ipynb
|-- out
|-- run2.sh
`-- src
    |-- __init__.py
    |-- collaborative_filtering.py
    |-- datafactory.py
    |-- proc_Flickr.py
    |-- proc_Foursquare.py
    |-- recommender.py
    |-- run1.py
    |-- run1_co.py
    |-- run1_rb.py
    |-- run2.py
    |-- runx.py
    |-- runx_batch.py
    |-- sim.py
    |-- ubm.py
    `-- utils.py
```

0. (c). Run:
```
python3 src/proc_Flickr.py
python3 src/proc_Foursquare.py
```
It will process source data, separate local residents from tourists, 
and apply Core-filtering.

1. Run:
```
python3 -u src/run1.py --city Rome --seed_list 2025 2026 2027 2028 2029
```
It will process user preferences in Rome city for five different data partition seeds 
and save results (`out/` dir) and corresponding training logs (`log/` dir) to disk. Change Rome to
Florence, Pisa, Istanbul, and London (one at a time) in ordred to process other cities.

2. Run:
```
./run2.sh Rome
```
It will estimate awareness set for each user, calibrate 
multinomial choice model, and save results (`out/` dir) and corresponding 
training logs (`log/` dir) to disk. Change Rome to Florence, Pisa, Istanbul, and London (one at a time) 
in ordred to process other cities.

3. Run:
```
python3 -u src/runx_batch.py --city Rome --seed_list 2025 2026 2027 2028 2029
```
This step will produce both: experiment artefacts (`.pk` files in `out/experiments/` dir) and experiment logs (`log/` dir). 
Change Rome to Florence, Pisa, Istanbul, and London (one at a time) in ordred to process other cities.


4. Run `notebooks/plot.ipynb` to plot (some of) figures from the original paper.
