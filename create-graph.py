#!/usr/bin/env python

"""
    create-graph.py
"""

from __future__ import print_function

import os
import sys
import h5py
import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

# --
# Helpers

def load_road_network(map_path='./data/manhattan.h5'):
    """
        load road network -- possibly over network, if local file isn't present
        
        !! osmnx can be a pain to install -- and not sure it works with python2.7
    """
    if not os.path.exists(map_path):
        import osmnx as ox
        G = ox.graph_from_place('Manhattan Island, New York City, New York, USA', network_type='drive')
        
        edges = np.array(G.edges)[:,:2]
        node_ids = np.array(G.nodes())
        node_coords = np.array([[data['y'], data['x']] for _, data in G.nodes(data=True)])
        
        f = h5py.File(map_path)
        f['edges'] = edges
        f['node_ids'] = node_ids
        f['node_coords'] = node_coords
        f.close()
    else:
        f = h5py.File(map_path)
        edges = f['edges'].value
        node_ids = f['node_ids'].value
        node_coords = f['node_coords'].value
        f.close()
    
    return {
        "edges" : edges,
        "node_ids" : node_ids,
        "node_coords" : node_coords,
    }


def latlon2cartesian(latlon, d=1):
    latlon = np.radians(latlon)
    return np.array([
        d * np.cos(latlon[:,0]) * np.cos(-latlon[:,1]), # x
        d * np.cos(latlon[:,0]) * np.sin(-latlon[:,1]), # y
        d * np.sin(latlon[:,0]),                        # z
    ]).T


def get_nearest_nodes(road_network, query_coords):
    # !! Should sometimes be worried about underflow
    
    node_coords_c = latlon2cartesian(road_network['node_coords'])
    query_coords_c = latlon2cartesian(query_coords)
    
    kd_tree = KDTree(node_coords_c, 2, metric='euclidean')
    dists, nns = kd_tree.query(query_coords_c)
    return road_network['node_ids'][nns], np.arcsin(dists.squeeze().clip(max=1)) * 6371 * 1000

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ride-path', type=str, default='./data/sunday.tsv')
    parser.add_argument('--map-path', type=str, default='./data/manhattan.h5')
    parser.add_argument('--target-date', type=str, default='2011-06-26')
    parser.add_argument('--distance-threshold', type=float, default=250)
    return parser.parse_args()

if __name__ == "__main__":
    # --
    # IO

    args = parse_args()

    # Load taxi data
    df = pd.read_csv(args.ride_path, sep='\t')
    road_network = load_road_network(args.map_path)

    # Map each ride to NN in graph
    df['pickup_nearest_node'], df['pickup_nearest_dist'] =\
        get_nearest_nodes(road_network, np.array(df[['pickup_latitude', 'pickup_longitude']]))

    df['dropoff_nearest_node'], df['dropoff_nearest_dist'] =\
        get_nearest_nodes(road_network, np.array(df[['dropoff_latitude', 'dropoff_longitude']]))

    # Drop rides that are too far from network
    df = df[(df.pickup_nearest_dist < args.distance_threshold) & (df.dropoff_nearest_dist < args.distance_threshold)]

    # Create nodes in spacetime graph
    df['pickup_node_spacetime']  = df.pickup_nearest_node.astype(str) + "-" + df.hour.astype(str)
    df['dropoff_node_spacetime'] = df.dropoff_nearest_node.astype(str) + "-" + df.hour.astype(str)

    # Count rides for foreground/background (eg date of interest vs nearby dates at same time/location)
    foreground_sel = (df.date == args.target_date)
    background, foreground = df[~foreground_sel], df[foreground_sel]

    n_background_days = background.date.unique().shape[0]
    nodes = pd.DataFrame({
        "fg_pickup"  : foreground.groupby('pickup_node_spacetime').date.count(),
        "fg_dropoff" : foreground.groupby('dropoff_node_spacetime').date.count(),
        "bg_pickup"  : background.groupby('pickup_node_spacetime').date.count() / n_background_days,
        "bg_dropoff" : background.groupby('dropoff_node_spacetime').date.count() / n_background_days,
    }).fillna(0)

    nodes['fg'] = nodes.fg_pickup + nodes.fg_dropoff
    nodes['bg'] = nodes.bg_pickup + nodes.bg_dropoff
    nodes['difference'] = nodes.fg - nodes.bg

    # Make spacetime graph -- geo graph replicated in time
    edges_str = pd.DataFrame(road_network['edges']).astype(str)
    edges_str = edges_str.drop_duplicates().reset_index(drop=True)
    space_edges = [(edges_str + '-%d' % i) for i in range(0, 24)]

    nodes_str = pd.Series(road_network['node_ids']).astype(str)
    time_edges = [pd.DataFrame({
        0 : (nodes_str + '-%d' % i),
        1 : (nodes_str + '-%d' % (i - 1)),
    }) for i in range(1, 24)]

    spacetime_edges = pd.concat(space_edges + time_edges).sort_values(0)
    spacetime_edges.columns = ('src', 'trg')

    # Add observations for all nodes in network
    nodes = nodes\
        .reindex(set(np.unique(spacetime_edges)))\
        .fillna(0)\
        .reset_index()

    # Save edges + nodes
    spacetime_edges.to_csv('./data/sunday-edges.tsv', sep='\t', index=False)
    nodes.to_csv('./data/sunday-nodes.tsv', sep='\t', index=False)

    # Expand `node_coords` to match ordering of `nodes`
    node_id_lookup = pd.Series(np.arange(road_network['node_ids'].shape[0]), index=road_network['node_ids'])
    coords = pd.Series(nodes['index']).apply(lambda x: x.split('-')[0])
    coords = np.array(node_id_lookup.loc[np.array(coords).astype(int)])
    coords = road_network['node_coords'][coords]
    np.save('./data/sunday-coords', coords)

