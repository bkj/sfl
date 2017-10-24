#!/usr/bin/env python

"""
    supervx.py
    
    Classes for preprocessing a problem before passing to SnapVX
    
    SnapVX has a "useClustering" option, but it doesn't vectorize the variables like we do here
    Using this wrapper instead of "out of the box" SnapVX on the taxi problem gives you a
    15-20x speedup.
"""

from __future__ import print_function

import sys
import numpy as np
from collections import defaultdict

from snapvx import *
import networkx as nx
from community import community_louvain

from time import time

class SuperVX(object):
    
    def __init__(self, supernodes, superedges, reg_sparsity=5, reg_edge=10):
        self.reg_sparsity = reg_sparsity
        self.reg_edge = reg_edge
        
        self.gvx = self._make_gvx(supernodes, superedges)
        self.values = self._add_objectives(self.gvx, supernodes, superedges)
                
    def _make_gvx(self, supernodes, superedges):
        # Construct supergraph
        sgraph = PUNGraph.New()
        _ = [sgraph.AddNode(l) for l in supernodes.keys()]
        _ = [sgraph.AddEdge(e1, e2) for e1, e2 in superedges.keys() if e1 != e2]
        return TGraphVX(sgraph)
    
    def _add_objectives(self, gvx, supernodes, superedges):
        # Node objectives
        values = {}
        for sid, sfeats in supernodes.items():
            v = Variable(sfeats.shape[0], name='x')
            int_edges = np.array(superedges[sid, sid])
            gvx.SetNodeObjective(sid, (
                0.5 * sum_squares(v - sfeats) + 
                self.reg_sparsity * norm1(v) + 
                self.reg_edge * norm1(v[int_edges[:,0]] - v[int_edges[:,1]])
            ))
            values[sid] = v
        
        # Edge objectives
        for (sid1, sid2), ext_edges in superedges.items():
            if sid1 != sid2:
                ext_edges = np.array(superedges[sid1, sid2])
                gvx.SetEdgeObjective(sid1, sid2, (
                self.reg_edge * norm1(values[sid1][ext_edges[:,0]] - values[sid2][ext_edges[:,1]])
            ))
        
        return values
    
    def solve(self, **kwargs):
        return self.gvx.Solve(**kwargs)


class SuperGraph(object):
    
    def __init__(self, edges, feats, partition=None):
        
        # Partition graph
        G = nx.from_edgelist(edges)
        if partition is None:
            t = time()
            print("SuperGraph: computing partition", file=sys.stderr)
            partition = community_louvain.best_partition(G)
            partition = np.array([p[1] for p in sorted(partition.items(), key=lambda x: x[0])])
            print("    took %fs" % (time() - t), file=sys.stderr)
        
        self.lookup, self.supernodes = self._make_supernodes(feats, partition)
        self.superedges = self._make_superedges(G, partition, self.lookup)
        self.partition = partition
        
    def _make_supernodes(self, feats, partition):
        lookup = {}
        supernodes = {}
        for p in np.unique(partition):
            psel = np.where(partition == p)[0]
            lookup[p] = dict(zip(psel, range(len(psel))))
            supernodes[p] = feats[psel]
        
        return lookup, supernodes
    
    def _make_superedges(self, G, partition, lookup):
        superedges = defaultdict(list)
        for node in G.nodes():
            p_node = partition[node]
            for neib in G.neighbors(node):
                p_neib = partition[neib]
                
                if (p_node == p_neib) and (node >= neib):
                    continue
                elif p_node > p_neib:
                    continue
                
                superedge = (lookup[p_node][node], lookup[p_neib][neib])
                superedges[(p_node, p_neib)].append(superedge)
        
        return superedges
    
    def unpack(self, values):
        values = dict([(k, np.asarray(v.value).squeeze()) for k,v in values.items()])
        
        reverse_lookup = [[(i[0], (sk, i[1])) for i in v.items()] for sk,v in self.lookup.items()]
        reverse_lookup = dict(reduce(lambda a,b: a + b, reverse_lookup))
        
        unpacked = np.zeros(len(reverse_lookup))
        for idx, (p, int_idx) in reverse_lookup.items():
            unpacked[idx] = values[p][int_idx]
        
        return unpacked