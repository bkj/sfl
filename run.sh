#!/bin/bash

# run.sh

# --
# Download

mkdir -p ./data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2011-05.csv
wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2011-06.csv
mv yellow* ./data

# --
# Preprocessing

# Pull out dates/times we care about
python subset-data.py

# Create spacetime graph
python create-graph.py

# --
# Run sparse fused lasso
# (Plots are saved to `results.png`)

# Simplest example
python sfl.py --savefig

# Run w/ ADMM on a small dataset
python sfl.py --savefig --use-admm

# Run w/ ADMM on datasets of increasing (temporal) size
python sfl.py --savefig --use-admm --time-window 8,16
python sfl.py --savefig --use-admm --time-window 4,20

# Run w/ ADMM on entire dataset
python sfl.py --savefig --use-admm --time-window 0,24
