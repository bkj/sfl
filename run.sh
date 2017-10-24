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