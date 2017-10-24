# Copyright (c) 2015, Stanford University. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import print_function

from snap import *
from cvxpy import *

import os
import sys
import json
from time import time
import numpy as np
import multiprocessing
from functools import partial
from scipy.sparse import lil_matrix

import __builtin__

# --
# Helpers

def robust_solve(problem):
    try:
        problem.solve()
    except SolverError:
        print("ECOS error: using SCS for x update (1)", file=sys.stderr)
        problem.solve(solver=SCS)
    
    if problem.status in [INFEASIBLE_INACCURATE, UNBOUNDED_INACCURATE]:
        print("ECOS error: using SCS for x update (2)", file=sys.stderr)
        problem.solve(solver=SCS)


class TGraphVX(TUNGraph):
    
    __default_objective = norm(0)
    __default_constraints = []
    
    def __init__(self, Graph=None):
        
        self.node_objectives = {}
        self.node_variables = {}
        self.node_constraints = {}
        self.edge_objectives = {}
        self.edge_constraints = {}
        self.node_values = {}
        self.all_variables = set()
        self.status = None
        self.value = None
        
        nodes = 0
        edges = 0
        if Graph != None:
            nodes = Graph.GetNodes()
            edges = Graph.GetEdges()
        
        TUNGraph.__init__(self, nodes, edges)
        
        if Graph != None:
            for ni in Graph.Nodes():
                self.AddNode(ni.GetId())
            for ei in Graph.Edges():
                self.AddEdge(ei.GetSrcNId(), ei.GetDstNId())
    
    def Nodes(self):
        ni = TUNGraph.BegNI(self)
        for i in xrange(TUNGraph.GetNodes(self)):
            yield ni
            ni.Next()
    
    def Edges(self):
        ei = TUNGraph.BegEI(self)
        for i in xrange(TUNGraph.GetEdges(self)):
            yield ei
            ei.Next()
    
    def Solve(self, UseADMM=True, NumProcessors=0, Rho=1.0, MaxIters=250, EpsAbs=0.01, 
        EpsRel=0.01, Verbose=False, UseClustering=False, ClusterSize=1000):
        
        if UseADMM and self.GetEdges() != 0:
            self.__SolveADMM(NumProcessors, Rho, MaxIters, EpsAbs, EpsRel, Verbose)
        else:
            self.__SerialADMM()
    
    def __SerialADMM(self):
        objective = 0
        constraints = []
        for ni in self.Nodes():
            nid = ni.GetId()
            objective += self.node_objectives[nid]
            constraints += self.node_constraints[nid]
        
        for ei in self.Edges():
            etup = self.__GetEdgeTup(ei.GetSrcNId(), ei.GetDstNId())
            objective += self.edge_objectives[etup]
            constraints += self.edge_constraints[etup]
        
        objective = Minimize(objective)
        problem = Problem(objective, constraints)
        
        robust_solve(problem)
        
        self.status = problem.status
        self.value = problem.value
        
        for ni in self.Nodes():
            nid = ni.GetId()
            variables = self.node_variables[nid]
            value = None
            for (varID, varName, var, offset) in variables:
                if var.size[0] == 1:
                    val = np.array([var.value])
                else:
                    val = np.array(var.value).reshape(-1,)
                if value is None:
                    value = val
                else:
                    value = np.concatenate((value, val))
            
            self.node_values[nid] = value
    
    def __SolveADMM(self, numProcessors, rho, maxIters, eps_abs, eps_rel, verbose):
        global node_vals, edge_z_vals, edge_u_vals
        global getValue
        
        manager = multiprocessing.Manager()
        node_vals = manager.dict()
        edge_z_vals = manager.dict()
        edge_u_vals = manager.dict()
        
        num_processors = multiprocessing.cpu_count() if numProcessors <= 0 else numProcessors
        
        # --
        # Set up nodes
        
        node_info = {}
        n_nodevars = 0
        for ni in self.Nodes():
            nid = ni.GetId()
            
            objectives  = self.node_objectives[nid]
            variables   = self.node_variables[nid]
            constraints = self.node_constraints[nid]
            
            neighbor_ids = [ni.GetNbrNId(j) for j in xrange(ni.GetDeg())]
            for neighbor_id in neighbor_ids:
                etup = self.__GetEdgeTup(nid, neighbor_id)
                constraints += self.edge_constraints[etup]
            
            size = sum([var.size[0] for (_, _, var, _) in variables])
            node_info[nid] = {
                "nid"          : nid,
                "objectives"   : objectives,
                "variables"    : variables,
                "constraints"  : constraints,
                "idx"          : n_nodevars,
                "size"         : size,
                "neighbor_ids" : neighbor_ids,
                "edges"        : [],
            }
            n_nodevars += size
        
        # --
        # Set up edges
        
        edge_list = []
        edge_info = {}
        n_edgevars = 0
        for ei in self.Edges():
            etup = self.__GetEdgeTup(ei.GetSrcNId(), ei.GetDstNId())
            
            node_i = node_info[etup[0]]
            node_j = node_info[etup[1]]
            
            idx_zij = n_edgevars
            idx_uij = n_edgevars
            n_edgevars += node_i["size"]
            
            idx_zji = n_edgevars
            idx_uji = n_edgevars
            n_edgevars += node_j["size"]
            
            tup = {
                "eid"         : etup,
                "objectives"  : self.edge_objectives[etup],
                "constraints" : (
                    self.node_constraints[etup[0]] +
                    self.node_constraints[etup[1]] +
                    self.edge_constraints[etup]
                ),
                
                "vars_i"  : node_i["variables"],
                "size_i"  : node_i["size"],
                "idx_i"   : node_i["idx"],
                "idx_zij" : idx_zij,
                "idx_uij" : idx_uij,
                
                "vars_j"  : node_j["variables"],
                "size_j"  : node_j["size"],
                "idx_j"   : node_j["idx"],
                "idx_zji" : idx_zji,
                "idx_uji" : idx_uji,
            }
            edge_list.append(tup)
            edge_info[etup] = tup
        
        A = lil_matrix((n_edgevars, n_nodevars), dtype=np.int8)
        for ei in self.Edges():
            etup = self.__GetEdgeTup(ei.GetSrcNId(), ei.GetDstNId())
            info_edge = edge_info[etup]
            node_i = node_info[etup[0]]
            node_j = node_info[etup[1]]
            
            for offset in xrange(node_i["size"]):
                row = info_edge["idx_zij"] + offset
                col = node_i["idx"] + offset
                A[row, col] = 1
            
            for offset in xrange(node_j["size"]):
                row = info_edge["idx_zji"] + offset
                col = node_j["idx"] + offset
                A[row, col] = 1
        
        A_tr = A.transpose()
        
        node_list = []
        for nid, entry in node_info.iteritems():
            for neighbor_id in entry["neighbor_ids"]:
                indices = ("idx_zij", "idx_uij") if nid < neighbor_id else ("idx_zji", "idx_uji")
                einfo = edge_info[self.__GetEdgeTup(nid, neighbor_id)]
                entry['edges'].append((
                    einfo[indices[0]],
                    einfo[indices[1]]
                ))
            
            node_list.append(entry)
        
        t = time()
        pool = multiprocessing.Pool(num_processors)
        edge_z_old = None
        for iter_ in range(maxIters):
            pool.map(partial(ADMM_x, rho=rho), node_list)
            pool.map(partial(ADMM_z, rho=rho), edge_list)
            pool.map(ADMM_u, edge_list)
            
            stats, stop, edge_z_old = self.__CheckConvergence(A, A_tr, edge_z_old, rho, eps_abs, eps_rel)
            stats.update({
                "iter" : iter_,
                "time" : time() - t,
            })
            print(json.dumps(stats))
            if stop:
                break
        
        pool.close()
        pool.join()
        
        # Clean up
        for entry in node_list:
            self.node_values[entry['nid']] = getValue(node_vals, entry['idx'], entry['size'])
        
        self.complete = iter_ <= maxIters
        self.value = self.GetTotalProblemValue()
    
    def __CheckConvergence(self, A, A_tr, edge_z_old, rho, e_abs, e_rel):
        node   = np.hstack([node_vals[k] for k in sorted(node_vals.keys())])
        edge_z = np.hstack([edge_z_vals[k] for k in sorted(edge_z_vals.keys())])
        edge_u = np.hstack([edge_u_vals[k] for k in sorted(edge_u_vals.keys())])
        
        Ax = A.dot(node)
        if edge_z_old is not None:
            s = rho * A_tr.dot(edge_z - edge_z_old)
        else:
            s = rho * A_tr.dot(edge_z)
        
        e_pri = np.sqrt(A.shape[1]) * e_abs + e_rel * max(np.linalg.norm(Ax), np.linalg.norm(edge_z)) + .0001
        e_dual = np.sqrt(A.shape[0]) * e_abs + e_rel * np.linalg.norm(rho * A_tr.dot(edge_u)) + .0001
        
        res_pri = np.linalg.norm(Ax - edge_z)
        res_dual = np.linalg.norm(s)
        
        return {
            "res_pri"  : res_pri,
            "e_pri"    : e_pri,
            "res_dual" : res_dual,
            "e_dual"   : e_dual,
        }, (res_pri <= e_pri) and (res_dual <= e_dual), edge_z
    
    def GetTotalProblemValue(self):
        result = 0.0
        for ni in self.Nodes():
            nid = ni.GetId()
            for (varID, varName, var, offset) in self.node_variables[nid]:
                var.value = self.GetNodeValue(nid, varName)
        
        for ni in self.Nodes():
            result += self.node_objectives[ni.GetId()].value
        
        for ei in self.Edges():
            etup = self.__GetEdgeTup(ei.GetSrcNId(), ei.GetDstNId())
            result += self.edge_objectives[etup].value
        
        return result
        
    # API to get node Variable value after solving with ADMM.
    def GetNodeValue(self, NId, Name):
        self.__VerifyNId(NId)
        for (varID, varName, var, offset) in self.node_variables[NId]:
            if varName == Name:
                offset = offset
                value = self.node_values[NId]
                return value[offset:(offset + var.size[0])]
        return None
        
    def __VerifyNId(self, NId):
        if not TUNGraph.IsNode(self, NId):
            raise Exception('Node %d does not exist.' % NId)
            
    def __UpdateAllVariables(self, NId, Objective):
        if NId in self.node_objectives:
            # First, remove the Variables from the old Objective.
            old_obj = self.node_objectives[NId]
            self.all_variables = self.all_variables - set(old_obj.variables())
        # Check that the Variables of the new Objective are not currently
        # in other Objectives.
        new_variables = set(Objective.variables())
        if __builtin__.len(self.all_variables.intersection(new_variables)) != 0:
            raise Exception('Objective at NId %d shares a variable.' % NId)
        self.all_variables = self.all_variables | new_variables
        
    def __ExtractVariableList(self, Objective):
        l = [(var.name(), var) for var in Objective.variables()]
        # Sort in ascending order by name
        l.sort(key=lambda t: t[0])
        l2 = []
        offset = 0
        for (varName, var) in l:
            # Add tuples of the form (id, name, object, offset)
            l2.append((var.id, varName, var, offset))
            offset += var.size[0]
        return l2
        
    def AddNode(self, NId, Objective=__default_objective, Constraints=__default_constraints):
        self.__UpdateAllVariables(NId, Objective)
        self.node_objectives[NId] = Objective
        self.node_variables[NId] = self.__ExtractVariableList(Objective)
        self.node_constraints[NId] = Constraints
        return TUNGraph.AddNode(self, NId)
        
    def SetNodeObjective(self, NId, Objective):
        self.__VerifyNId(NId)
        self.__UpdateAllVariables(NId, Objective)
        self.node_objectives[NId] = Objective
        self.node_variables[NId] = self.__ExtractVariableList(Objective)
        
    def GetNodeObjective(self, NId):
        self.__VerifyNId(NId)
        return self.node_objectives[NId]
        
    def SetNodeConstraints(self, NId, Constraints):
        self.__VerifyNId(NId)
        self.node_constraints[NId] = Constraints
        
    def GetNodeConstraints(self, NId):
        self.__VerifyNId(NId)
        return self.node_constraints[NId]
    
    def __GetEdgeTup(self, NId1, NId2):
        return (NId1, NId2) if NId1 < NId2 else (NId2, NId1)
    
    def __VerifyEdgeTup(self, ETup):
        if not TUNGraph.IsEdge(self, ETup[0], ETup[1]):
            raise Exception('Edge {%d,%d} does not exist.' % ETup)
    
    def AddEdge(self, SrcNId, DstNId, ObjectiveFunc=None,
            Objective=__default_objective, Constraints=__default_constraints):
        ETup = self.__GetEdgeTup(SrcNId, DstNId)
        if ObjectiveFunc != None:
            src_vars = self.GetNodeVariables(SrcNId)
            dst_vars = self.GetNodeVariables(DstNId)
            ret = ObjectiveFunc(src_vars, dst_vars)
            if type(ret) is tuple:
                self.edge_objectives[ETup] = ret[0]
                self.edge_constraints[ETup] = ret[1]
            else:
                self.edge_objectives[ETup] = ret
                self.edge_constraints[ETup] = self.__default_constraints
        else:
            self.edge_objectives[ETup] = Objective
            self.edge_constraints[ETup] = Constraints
        return TUNGraph.AddEdge(self, SrcNId, DstNId)
        
    def SetEdgeObjective(self, SrcNId, DstNId, Objective):
        ETup = self.__GetEdgeTup(SrcNId, DstNId)
        self.__VerifyEdgeTup(ETup)
        self.edge_objectives[ETup] = Objective
        
    def GetEdgeObjective(self, SrcNId, DstNId):
        ETup = self.__GetEdgeTup(SrcNId, DstNId)
        self.__VerifyEdgeTup(ETup)
        return self.edge_objectives[ETup]
        
    def SetEdgeConstraints(self, SrcNId, DstNId, Constraints):
        ETup = self.__GetEdgeTup(SrcNId, DstNId)
        self.__VerifyEdgeTup(ETup)
        self.edge_constraints[ETup] = Constraints
        
    def GetEdgeConstraints(self, SrcNId, DstNId):
        ETup = self.__GetEdgeTup(SrcNId, DstNId)
        self.__VerifyEdgeTup(ETup)
        return self.edge_constraints[ETup]
    
    def GetNodeVariables(self, NId):
        self.__VerifyNId(NId)
        d = {}
        for (varID, varName, var, offset) in self.node_variables[NId]:
            d[varName] = var
        
        return d



## =================================== ## 
## ADMM Global Variables and Functions ##

def getValue(shared, idx, length):
    try:
        return np.array(shared[idx])
    except:
        return np.zeros(length)


def setValue(shared, idx, val):
    shared[idx] = np.asarray(val).squeeze()


def writeObjective(shared, idx, objective, variables):
    for v in objective.variables():
        for (varID, varName, var, offset) in variables:
            if varID == v.id:
                setValue(shared, idx + offset, v.value)
                break


def ADMM_x(node, rho):
    norms = 0
    
    for zi, ui in node["edges"]:
        for (varID, varName, var, offset) in node["variables"]:
            z = getValue(edge_z_vals, zi + offset, var.size[0])
            u = getValue(edge_u_vals, ui + offset, var.size[0])
            norms += square(norm(var - z + u))
    
    objective = Minimize(node["objectives"] + (rho / 2) * norms)
    problem = Problem(objective, node["constraints"])
    robust_solve(problem)
    
    writeObjective(node_vals, node["idx"], objective, node["variables"])
    return None


def ADMM_z(edge, rho):
    norms = 0
    
    for (varID, varName, var, offset) in edge["vars_i"]:
        x_i = getValue(node_vals, edge["idx_i"] + offset, var.size[0])
        u_ij = getValue(edge_u_vals, edge["idx_uij"] + offset, var.size[0])
        norms += square(norm(x_i - var + u_ij))
    
    for (varID, varName, var, offset) in edge["vars_j"]:
        x_j = getValue(node_vals, edge["idx_j"] + offset, var.size[0])
        u_ji = getValue(edge_u_vals, edge["idx_uji"] + offset, var.size[0])
        norms += square(norm(x_j - var + u_ji))
    
    objective = Minimize(edge["objectives"] + (rho / 2) * norms)
    problem = Problem(objective, edge["constraints"])
    robust_solve(problem)
    
    writeObjective(edge_z_vals, edge["idx_zij"], objective, edge["vars_i"])
    writeObjective(edge_z_vals, edge["idx_zji"], objective, edge["vars_j"])


def ADMM_u(edge):
    setValue(edge_u_vals, edge["idx_uij"], (
        getValue(edge_u_vals, edge["idx_uij"], edge["size_i"]) +
        getValue(node_vals, edge["idx_i"], edge["size_i"]) -
        getValue(edge_z_vals, edge["idx_zij"], edge["size_i"])
    ))
    
    setValue(edge_u_vals, edge["idx_uji"], (
        getValue(edge_u_vals, edge["idx_uji"], edge["size_j"]) +
        getValue(node_vals, edge["idx_j"], edge["size_j"]) -
        getValue(edge_z_vals, edge["idx_zji"], edge["size_j"])
    ))
