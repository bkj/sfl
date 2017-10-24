#!/usr/bin/env python

"""
    sfl.py
"""

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import sys
import argparse
import numpy as np
import pandas as pd

from supervx import SuperVX, SuperGraph

# --
# Helpers

def filter_time(nodes, edges, coords, min_hour, max_hour):
    """
        Sometimes we want to run on only a subset of times
    """
    hrs = map(str, range(min_hour, max_hour + 1))
    
    # Edges in time period (pickup and dropoff)
    edges = edges[(
        edges.src.apply(lambda x: x.split('-')[1] in hrs) & 
        edges.trg.apply(lambda x: x.split('-')[1] in hrs)
    )].reset_index(drop=True)
    
    # Nodes + coords in time period
    sel = np.array(nodes['index'].apply(lambda x: x.split('-')[1] in hrs))
    nodes = nodes[sel].reset_index(drop=True)
    coords = coords[sel]
    
    return nodes, edges, coords


def make_edges_sequential(nodes, edges):
    """
        SnapVX requires/prefers nodes to have sequential IDs, w/o any gaps
    """
    node_lookup = pd.Series(np.arange(nodes.shape[0]), index=nodes['index'])
    
    edges = np.vstack([
        np.array(node_lookup.loc[edges.src]), 
        np.array(node_lookup.loc[edges.trg]),
    ]).T
    
    # Order edges + remove duplicates
    sel = edges[:,0] >= edges[:,1]
    edges[sel] = edges[sel,::-1]
    edges = np.vstack(set(map(tuple, edges)))
    
    return edges

# --
# Args

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--node-path', type=str, default='./data/sunday-nodes.tsv')
    parser.add_argument('--edge-path', type=str, default='./data/sunday-edges.tsv')
    parser.add_argument('--coord-path', type=str, default='./data/sunday-coords.npy')
    
    parser.add_argument('--time-window', type=str, default='12,13')
    
    parser.add_argument('--use-admm', action="store_true")
    parser.add_argument('--reg-sparsity', type=float, default=3)
    parser.add_argument('--reg-edge', type=float, default=6)
    
    parser.add_argument('--savefig', action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print('load data', file=sys.stderr)
    nodes  = pd.read_csv(args.node_path, sep='\t')
    edges  = pd.read_csv(args.edge_path, sep='\t')
    coords = np.load(args.coord_path)

    # --
    # Subset by time (maybe)
    
    if args.time_window:
        print('filter time', file=sys.stderr)
        min_hour, max_hour = map(int, args.time_window.split(','))
        nodes, edges, coords = filter_time(nodes, edges, coords, min_hour, max_hour)
    
    # --
    # Convert IDs to sequential ints
    
    print('sequential ids', file=sys.stderr)
    sequential_edges = make_edges_sequential(nodes, edges)
    targets = np.array(nodes.difference)
    
    # --
    # Run sparse fused lasso
    
    partition = None if args.use_admm else np.ones(nodes.shape[0]).astype(int)
    supergraph = SuperGraph(sequential_edges, targets, partition=partition)
    
    supervx = SuperVX(
        supergraph.supernodes,
        supergraph.superedges,
        reg_sparsity=args.reg_sparsity,
        reg_edge=args.reg_edge
    )

    _ = supervx.solve(
        UseADMM=args.use_admm,
        Verbose=True
    )
    
    fitted = supergraph.unpack(supervx.values)
    print('done', file=sys.stderr)
    
    # --
    # Plot results (maybe)
    
    if args.savefig:
        print('plotting', file=sys.stderr)
        max_col = np.sqrt(np.abs(fitted)).max()
        hours = np.array(nodes['index'].apply(lambda x: x.split('-')[1])).astype('int')
        
        uhours = np.unique(hours)
        
        grid_dim = np.ceil(np.sqrt(len(uhours))).astype(int)
        f, axs = plt.subplots(grid_dim, grid_dim, sharex='col', sharey='row', figsize=(15, 15))
        
        axs = np.hstack(axs)
        _ = [ax.axis('off') for ax in axs]
        for i, hour in enumerate(uhours):
            
            sel = hours == hour
            coords_ = coords[sel]
            fitted_ = fitted[sel]
            
            _ = axs[i].scatter(coords_[:,1], coords_[:,0], 
                c=np.sign(fitted_) * np.sqrt(np.abs(fitted_)), 
                vmin=-max_col, 
                vmax=max_col + 1, 
                s=1, 
                cmap='seismic'
            )
            _ = axs[i].set_title('hour=%d' % hour)
            
        plt.savefig('./results.png')
