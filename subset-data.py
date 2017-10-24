#!/usr/bin/env python

"""
    subset-data.py
    
    Pulls dates/times of interest from larger dataset.
    In particular, we want to get data from Sundays only.
"""

from __future__ import print_function

import pandas as pd
from glob import glob

# --
# Helpers

def parse_dublin(bbox_str):
    bbox = map(lambda x: float(x.split('=')[1]), bbox_str.split(';'))
    return dict(zip(['west', 'south', 'east', 'north'], bbox))

def filter_dublin(bbox_str, lat, lon):
    bbox = parse_dublin(bbox_str)
    return (
        (lat > bbox['south']) & 
        (lat < bbox['north']) & 
        (lon < bbox['east']) & 
        (lon > bbox['west'])
    )

def load_taxi(path):
    return pd.read_csv(path, usecols=[
        'pickup_datetime', 'dropoff_datetime', 
        'pickup_longitude', 'pickup_latitude', 
        'dropoff_longitude', 'dropoff_latitude',
    ], dtype={
        'pickup_datetime'   : str,
        'dropoff_datetime'  : str,
        'pickup_longitude'  : float,
        'pickup_latitude'   : float,
        'dropoff_longitude' : float,
        'dropoff_latitude'  : float,
    })


if __name__ == "__main__":
    print("loading")
    df = pd.concat(map(load_taxi, glob('./data/yellow*')))
    
    print("drop locations outside of NYC")
    bbox_str = "westlimit=-74.087334; southlimit=40.641833; eastlimit=-73.858681; northlimit=40.884708"
    df = df[filter_dublin(bbox_str, df.dropoff_latitude, df.dropoff_longitude)]
    df = df[filter_dublin(bbox_str, df.pickup_latitude, df.pickup_longitude)]
    
    print("cleaning dates")
    pickup_datetime = pd.to_datetime(df.pickup_datetime)
    df['date']      = df.pickup_datetime.apply(lambda x: x.split(' ')[0])
    df['dayofweek'] = pickup_datetime.dt.dayofweek
    df['hour']      = pickup_datetime.dt.hour
    
    print("Save data for Sundays in time period")
    sunday = df[(df.dayofweek == 6)]
    sunday.to_csv('./data/sunday.tsv', sep='\t', index=False)
