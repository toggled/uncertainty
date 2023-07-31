from src.graph import UGraph as Graph 
from copy import deepcopy,copy
import random 
from src.query import Query
from time import time 
import math 
from math import log2
import os,pickle
from scipy.stats import entropy
from datetime import datetime
import numpy as np
from src.utils import h 
import networkx as nx
from itertools import combinations,permutations
import json
from pptree import * 
from heapdict import heapdict 
from collections import deque 
import sys 

def save_pickle(ob, fname):
    with open (fname, 'wb') as f:
        #Use the dump function to convert Python objects into binary object files
        pickle.dump(ob, f)

def load_pickle(fname):
    with open (fname, 'rb') as f:
        #Convert binary object to Python object
        ob = pickle.load(f)
        return ob

def get_sample(G, seed):
    random.seed(seed)
    poss_world = Graph()
    for e in G.Edges:
        p = G.edict[e]
        if random.random() < p:
            poss_world.add_edge(e[0],e[1],p,G.weights[e],construct_nbr=True)
    # sample_tm = time() - start_execution_time
    # self.sample_time_list.append(sample_tm)
    # self.total_sample_tm += sample_tm
    return (poss_world,0) 

class Algorithm:
    """ A generic Algorithm class that implmenets all the Exact algorithms in the paper."""
    def __init__(self, g, query, debug = False) -> None:
        self.debug = debug
        self.algostat = {} 
        # self.G = deepcopy(g)
        self.G = g #
        assert isinstance(self.G, Graph)
        assert isinstance(query, Query)
        self.Query = query
        self.algostat['execution_time'] = 0
        self.algostat['result'] = {}
        self.algostat['k'] = 0
        self.algostat['algorithm'] = ''
        self.algostat['support'] = []
        # self.algostat['peak_memB'] = []

    def measure_uncertainty(self):
        """ 
        Measures entropy exactly by sampling all possible worlds. 
        """
        start_execution_time = time()
        self.Query.eval(dontenumerateworlds = True)
        tm2 = time()
        if self.Query.bucketing:
            H = self.Query.compute_bucketed_entropy() # Entropy under uniform bucketing strategy
        else: # un-bucketted entropy
            H = self.Query.compute_entropy()
        self.algostat['execution_time'] = time() - start_execution_time
        self.algostat['algorithm'] = 'exact'
        self.algostat['H'] = H 
        self.algostat['support'] =  str(list(self.Query.get_support()))
        self.algostat['query_eval_tm'] = sum(self.Query.evaluation_times) # time spent on query evlauation
        self.algostat['total_sample_tm'] = (tm2-start_execution_time) - self.algostat['query_eval_tm'] # Time spend on possible world sampling
        self.Query.clear()
        return H
         

    # Exhaustive search algorithm. All (m choose k) choices are explored. 
    def Bruteforce(self, k, update_type = 'o1',verbose = False):
        """ 
        Brute-force algorithm
        Returns selected edgeset, entropy value after selection, and entropy-reduction amount (DeltaH)
        """
        assert k>=1
        self.algostat['algorithm'] = 'exact'
        self.algostat['k'] = k
        start_execution_time = time()
        self.Global_maxima = None  
        self.Query.eval()
        H0 = self.Query.compute_entropy()
        self.algostat['result']['H0'] = H0
        for edge_set in self.G.enumerate_k_edges(k):
            if (verbose):
                print(edge_set)
            g_copy = deepcopy(self.G)
            for e in edge_set:
                g_copy.edge_update(e[0],e[1], type= update_type)

            self.Query.reset(g_copy)
            self.Query.eval()
            Hi = self.Query.compute_entropy()
            if self.Global_maxima is not None:
                if H0 -  Hi > H0 - self.Global_maxima[1]:
                    self.Global_maxima = (edge_set, Hi)
                    if (verbose):
                        print('new maxima: ',self.Global_maxima)
            else:
                self.Global_maxima = (edge_set, Hi)
                if (verbose):
                    print('Initial maxima: ',self.Global_maxima)
            # print('e: ',edge_set,' \Delta H = ', H0-Hi, ' H0:', H0, ' H_next: ',Hi)
        self.algostat['execution_time'] = time() - start_execution_time
        self.algostat['result']['edges'] = self.Global_maxima[0]
        self.algostat['result']['H*'] = self.Global_maxima[1]
        self.algostat['support'] = ''
        self.algostat['DeltaH'] = self.algostat['result']['H0'] - self.algostat['result']['H*']
        self.algostat['|DeltaH|'] = abs(self.algostat['result']['H0'] - self.algostat['result']['H*'])
        self.algostat['history_deltaH'] = [] # history is empty because it considers all possible choices of k-edges at once.
        return self.Global_maxima[0], self.Global_maxima[1],  H0 - self.Global_maxima[1]
        

    # def algorithm5(self, k, update_type = 'o1',verbose = False):
    #     """ 
    #     Exact contribution table version of Algorithm 5
    #     Returns selected edgeset, entropy value after selection, and entropy-reduction amount (DeltaH)
    #     """
    #     self.algostat['algorithm'] = 'Alg5'
    #     self.algostat['k'] = k
    #     self.Query.eval()
    #     self.algostat['result']['H0'] =  self.Query.compute_entropy() # Initial entropy. Kept outside of time() because H0 is not needed in contr table and computed only for logging.
        
    #     start_execution_time = time()
    #     self.Query.constructTables(op = update_type) 
    #     Estar = copy(self.G.edict)
    #     E = []
        
    #     for iter in range(k):
    #         # Start of Algorithm 4
    #         local_maxima = None 
    #         # Construct Pr[] and \DelPr[]
    #         Pr = {}
    #         DelPr = {}
    #         sum_logPrDelPr = 0 
    #         for omega in self.Query.phiInv:
    #             for i in self.Query.phiInv[omega]:
    #                 Pr[omega] = Pr.get(omega,0) + sum(self.Query.C[i].values())
    #                 DelPr[omega] = DelPr.get(omega,0) + sum(self.Query.DeltaC[i].values())
    #             if (Pr[omega] == 0):
    #                 sum_logPrDelPr += 0 
    #                 # for i in self.Query.phiInv[omega]:
    #                 #     print(i,self.Query.C[i].values())
    #             else:
    #                 sum_logPrDelPr += (math.log2(Pr[omega],2) * DelPr[omega])

    #         if (verbose):
    #             print('Selecting edge#: ',iter+1)
    #             print('sum logPr DelPr = ', -sum_logPrDelPr)
    #         for e in Estar: # Among remaining edges (Estar), find the one (e*) with largest reduction in Entropy
    #             delpe = self.G.get_next_prob(e[0],e[1],update_type) - self.G.edict[e]
    #             DeltaHe = (-1/delpe) * sum_logPrDelPr
    #             if verbose:
    #                 print('e: ',e,' DeltaH[e]: ',DeltaHe)
    #             if local_maxima:
    #                 if DeltaHe > local_maxima[1]:
    #                     local_maxima = (e,DeltaHe)
    #             else:
    #                local_maxima = (e,DeltaHe)
    #         estar = local_maxima[0] # assign e*
    #         # End of Algorithm 4
    #         E.append(estar)
    #         if (verbose):
    #             print('e* = ',estar)
    #         del Estar[estar] # Delete e* from dictionary s.t. next iteration is 1 less than the current.
    #         self.G.edge_update(estar[0],estar[1],type = update_type) # Update UGraph()
    #         self.Query.reset(self.G) # Re-initialise Query() with updated UGraph()
    #         if verbose:
    #             print('C: ',self.Query.C)
    #         self.Query.updateTables()
    #         if (verbose):
    #             print('C after p(e) = ',self.Query.C)
           
    #     self.algostat['execution_time'] = time() - start_execution_time
    #     self.algostat['result']['edges'] = E
    #     self.Query.eval()
    #     self.algostat['result']['H*'] = self.Query.compute_entropy()
    #     return E,self.algostat['result']['H*'],self.algostat['result']['H0']- self.algostat['result']['H*']

    # def calculate_Jacobian(self,op):
    #     """ 
    #     Calculate Jacobian and 
    #     """
    #     self.Jacobian = {}
    #     for omega in self.Query.phiInv:
    #         self.Jacobian[omega] = {}
    #         for e in self.G.Edges:
    #             p_e = self.G.edict[e] # p(e)
    #             p_up_e = self.G.get_next_prob(e[0],e[1],op) # p_up(e)
    #             _sum = 0
    #             for i in self.Query.phiInv[omega]:
    #                 if e in self.Query.C[i]: # if edge e is in G_i where omega is observed
    #                     if p_e != p_up_e:
    #                         G_up = self.Query.PrG[i] * p_up_e / p_e  # Pr_up(G_i) = Pr(G) * p_up(e) / p(e)
    #                         _sum += (G_up - self.Query.PrG[i])
    #                 else: #  edge e not in G_i where omega is observed
    #                     if p_e != p_up_e:
    #                         try:
    #                             G_up = self.Query.PrG[i] * (1-p_up_e) / (1-p_e)  # Pr_up(G_i) = Pr(G_i) * (1- p_up(e)) / (1- p(e))
    #                             _sum += (G_up - self.Query.PrG[i])
    #                         except ZeroDivisionError:
    #                             print(p_e,p_up_e,self.Query.PrG[i])
    #                             raise ZeroDivisionError
                                
    #             self.Jacobian[omega][e] = _sum 

    # def compute_Pr_up(self,op, verbose = False):
    #     """ 
    #     Calculate 
    #     Pr_up(G_i,e_j) (Pr_up(G_i) after p(e_j) is updated ) 
    #     """
    #     self.Pr_up = {}
    #     if (verbose):
    #         self.Gsums = {}
    #     for omega in self.Query.phiInv:
    #         self.Pr_up[omega] = {}
    #         if (verbose):
    #             self.Gsums[omega] = {}
    #         for e in self.G.Edges:
    #             p_e = self.G.edict[e] # p(e)
    #             p_up_e = self.G.get_next_prob(e[0],e[1],op) # p_up(e)
    #             _sumPrG = 0
    #             if (verbose):
    #                 self.Gsums[omega][e] = []
    #             for i in self.Query.phiInv[omega]:
    #                 if e in self.Query.index[i]: # if edge e is in G_i where omega is observed
    #                     if p_e != p_up_e:
    #                         G_up = self.Query.PrG[i] * p_up_e / p_e  # Pr_up(G_i) = Pr(G) * p_up(e) / p(e)
    #                         _sumPrG += G_up 
    #                         if (verbose):
    #                             self.Gsums[omega][e].append((i,1))
    #                 else: #  edge e not in G_i where omega is observed
    #                     if p_e != p_up_e:
    #                         try:
    #                             G_up = self.Query.PrG[i] * (1-p_up_e) / (1-p_e)  # Pr_up(G_i) = Pr(G_i) * (1- p_up(e)) / (1- p(e))
    #                             _sumPrG += G_up 
    #                             if (verbose):
    #                                 self.Gsums[omega][e].append((i,0))
    #                         except ZeroDivisionError:
    #                             print(p_e,p_up_e,self.Query.PrG[i])
    #                             raise ZeroDivisionError
                                
    #             self.Pr_up[omega][e] = _sumPrG

    # def algorithm5(self, k, update_type = 'o1',verbose = False):
    #     """ 
    #     Algorithm 5 (with Exact memoization)
    #     Returns selected edgeset, entropy value after selection, and entropy-reduction amount (DeltaH)
    #     """
    #     self.algostat['algorithm'] = 'greedy+mem(exact)'
    #     self.algostat['k'] = k
    #     self.Query.eval()
    #     self.algostat['result']['H0'] =  self.Query.compute_entropy() # Initial entropy. Kept outside of time() because H0 is not needed in contr table and computed only for logging.
    #     start_execution_time = time()
    #     self.Query.constructTables(op = update_type,verbose = verbose) 
    #     Estar = copy(self.G.edict)
    #     E = []
    #     if (verbose):
    #         print('H0: ',self.algostat['result']['H0'])
    #         print('p(e): ',self.G.edict)
    #         print('Pr[G]: ',self.Query.PrG)
    #         print('G: ',self.Query.G)
    #         # print('Pr[Omega]: ', self.Query.freq_distr)
    #         # print('results: ',self.Query.results)
    #     # Calculate Pr_up^{e_j}(G_i) for all i,j
    #     self.compute_Pr_up(update_type,verbose = verbose)
    #     for iter in range(k):
    #         if (verbose):
    #             print("Iteration: ",iter)
    #         # Start of Algorithm 4
            
    #         # Construct Pr[Omega] 
    #         Pr_Omega = {}
    #         for omega in self.Query.phiInv:
    #             Pr_Omega[omega] = sum([self.Query.PrG[i] for i in self.Query.phiInv[omega]])
    #         if self.debug: print('Pr[Omega]: ',Pr_Omega)
    #         if self.debug:   M = deepcopy(self.Pr_up)
    #         if self.debug:  print("M before update: \n\r",self.Pr_up)
    #         if (verbose):
    #             self.Query.eval()
    #             H0 = self.Query.compute_entropy()
    #             # print('H0: ',H0)

    #         if (verbose):
    #             print('Selecting edge#: ',iter+1)
    #             # print('Jacobian: ', DelPr_Omega_pe)
    #             print('Pr[Omega]: ',Pr_Omega)
    #             print("Pr_up[Omega]: ",self.Pr_up)
    #             print('Gsums: ',self.Gsums)
    #         local_maxima = None 
    #         for e in Estar: # Among remaining edges (Estar), find the one (e*) with largest  H - H_{up}
    #             DeltaHe2 = 0 # H - H^{e}_{up}
    #             for omega in self.Query.phiInv:
    #                 if (Pr_Omega[omega] != 0):
    #                     if (self.Pr_up[omega][e] != 0):
    #                         DeltaHe2 += (log2(self.Pr_up[omega][e])*self.Pr_up[omega][e] - log2(Pr_Omega[omega])*Pr_Omega[omega])
    #                     else:
    #                         DeltaHe2 += (0 - log2(Pr_Omega[omega])*Pr_Omega[omega])
    #                 else:
    #                     if (self.Pr_up[omega][e] != 0):
    #                         DeltaHe2 += (log2(self.Pr_up[omega][e])*self.Pr_up[omega][e] - 0)
    #                     else:
    #                         DeltaHe2 += 0

    #             if verbose:
    #                 print('e: ',e,' DeltaH2[e]: ',DeltaHe2)
    #                 print('H_next [',e,']: ', H0 + DeltaHe2)
                
    #             if local_maxima is not None:
    #                 if DeltaHe2 > local_maxima[1]:
    #                     local_maxima = (e,DeltaHe2)
    #             else:
    #                local_maxima = (e,DeltaHe2)
    #         estar = local_maxima[0] # assign e*
    #         # End of Algorithm 4
    #         E.append(estar)
    #         if (verbose):
    #             print('e* = ',estar)
    #         del Estar[estar] # Delete e* from dictionary s.t. next iteration is 1 less than the current.
    #         self.Query.updateTables(estar, update_type, self.Pr_up)
    #         self.G.edge_update(estar[0],estar[1],type = update_type) # Update UGraph()
    #         self.Query.reset(self.G) # Re-initialise Query() with updated UGraph()
    #         if self.debug:  
    #             self.compute_Pr_up(update_type, verbose = verbose)
    #             print('-----')
    #             print(self.Pr_up)
    #         if (verbose):
    #             print('After p(e) update: ')
    #             # print('C = ',self.Query.C)
    #         if (verbose):
    #             print('----------')
    #     self.algostat['execution_time'] = time() - start_execution_time
    #     self.algostat['result']['edges'] = E
    #     self.Query.eval()
    #     self.algostat['result']['H*'] = self.Query.compute_entropy()
    #     # self.algostat['support'] = ','.join([str(i) for i in self.Query.phiInv.keys()])
    #     # self.algostat['support'] = str(list(self.Query.phiInv.keys()))
    #     self.algostat['support'] = ''
    #     self.algostat['DeltaH'] = self.algostat['result']['H0'] - self.algostat['result']['H*']
    #     self.algostat['|DeltaH|'] = abs(self.algostat['result']['H0'] - self.algostat['result']['H*'])
    #     if self.debug: self.algostat['M'] = M
    #     return E,self.algostat['result']['H*'],self.algostat['result']['H0']- self.algostat['result']['H*']

class ApproximateAlgorithm:
    """ Implements Algorithm 2 and Approximate variants of Algorithms 3, and 5"""
    def __init__(self, g, query, debug = False) -> None:
        self.debug = debug
        self.algostat = {} 
        self.G = deepcopy(g) 
        assert isinstance(self.G, Graph)
        # assert isinstance(query, Query)
        self.Query = query
        self.algostat['execution_time'] = 0
        self.algostat['result'] = {}
        self.algostat['k'] = 0
        self.algostat['support'] = [] # Since all sup values may not be observed in the sample, it is good to record those observed.
        self.algostat['algorithm'] = ''
        # self.algostat['peak_memB'] = []
    
    def measure_uncertainty(self, N=1, T=10, seed = 1):
        """
        Alg 2 
        """
        precomp = {}
        if os.environ['precomp']:
            previous_omega_files = [f for f in os.listdir(os.environ['precomp']) if f.endswith('.pre')]
            # print(previous_omega_files)
            if len(previous_omega_files):
                for f in previous_omega_files:
                    # print('f = ',f)
                    n,t = f.split('.pre')[0].split('_')
                    n,t = int(n),int(t)
                    if n not in precomp:
                        precomp[n] = {}
                    if t not in precomp[n]:
                        precomp[n][t] = f
                maximum_T = max(precomp[0].keys())+1
                maximum_N = max(precomp.keys())+1
                # print(maximum_N,maximum_T, N, T)
                assert N <= maximum_N, "precomputed possible worlds are insufficient"
                assert T <= maximum_T, "precomputed possible worlds are insufficient"
        start_execution_time = time()
        query_evaluation_times = []
        hat_H_list = []
        support_observed = []
        sum_H = 0
        if len(precomp)==0: # no pre
            for i in range(N):
                if self.Query.bucketing: # buckeing & no-pre
                    support_observed = []
                    hat_Pr = {}
                    j = 0
                    if 'time_seed' in os.environ:
                        random.seed()
                        s = random.randint(0,1000000)
                    else:
                        s = seed + i
                    # print('seed: ',s)
                    for g in self.G.get_Ksample(T,seed=s):
                        start_tm = time()
                        # omega = self.Query.evalG(g[0])
                        if self.Query.qtype == 'reach':
                            _, _, _,omega = g[0].bfs_sample(self.Query.u,self.Query.v,seed = s,optimiseargs=None)
                        elif self.Query.qtype == 'sp':
                            _, _, _,omega = g[0].dijkstra_sample(self.Query.u,self.Query.v, seed = s,optimiseargs=None)
                        elif self.Query.qtype == 'tri':
                            G = nx.Graph()
                            G.add_edges_from(g[0].Edges)
                            omega = sum(nx.triangles(G).values()) / 3
                        query_evaluation_times.append(time()-start_tm)
                        support_observed.append(omega)
                        if os.environ['precomp']:
                            save_pickle(omega, os.path.join(os.environ['precomp'],str(i)+"_"+str(j)+".pre"))
                            j+=1
                        # g[0].clear()
                    #bucketing 
                    S = support_observed
                    # S = list(set(support_observed))
                    _min,_max = min(S),max(S)
                    if _min == _max:
                        index_ub = 1
                        self.buckets = [(_min,_max)]
                        bucket_distr = {0: 0}
                    else:
                        # delta = math.floor(math.sqrt(len(set(S))))
                        delta = math.floor(math.sqrt(T))
                        # print('max = ',_max,' min= ',_min,' delta = ',delta)
                        index_ub = math.ceil((_max-_min)/delta)
                        self.buckets = [(_min+i*delta, _min+(i+1)*delta) for i in range(index_ub)]
                        # print(self.buckets)
                        bucket_distr = {i: 0 for i in range(index_ub)}
                    for s in S:
                        omega_i = (1.0/T)
                        if len(bucket_distr) == 1:
                            bucket_distr[0] += omega_i 
                        else:
                            _index = math.floor((s - _min)/delta)
                            bucket_distr[_index] = bucket_distr.get(_index,0) + omega_i
                    
                    hat_Pr = bucket_distr
                    # print([(self.buckets[_index],bucket_distr[_index]) for _index in bucket_distr])
                    # print('--- ',hat_Pr)
                    # print('-- ', S)
                    # print('buckets = ',self.buckets)
                else: # non-bucketing & no-pre
                    hat_Pr = {}
                    # j = 0
                    if 'time_seed' in os.environ:
                        random.seed()
                        s = random.randint(0,1000000)
                    else:
                        s = seed + i
                    # func_obj = self.G.get_Ksample(T,seed=s)
                    # for g in func_obj:
                    for j in range(T):
                        g,_ = get_sample(self.G,seed=s)
                        # print(g[0].Edges)
                        start_tm = time()
                        # omega = self.Query.evalG(g)
                        if self.Query.qtype == 'reach':
                            if len(g.Edges) == 0:
                                omega = 0
                            else:
                                _, _, _,omega = g.bfs_sample(self.Query.u,self.Query.v,seed = s,optimiseargs=None)
                        elif self.Query.qtype == 'sp':
                            if len(g.Edges) == 0:
                                omega = 0
                            else:
                                _, _, _,omega = g.dijkstra_sample(self.Query.u,self.Query.v, seed = s,optimiseargs=None)
                        elif self.Query.qtype == 'tri':
                            if len(g.Edges)<3:
                                omega = 0
                            else:
                                G = nx.Graph()
                                G.add_edges_from(g.Edges)
                                omega = sum(nx.triangles(G).values()) / 3
                        g = None 
                        # _,_,_,omega = g[0].bfs_sample(self.Query.u,self.Query.v,optimiseargs=None)
                        query_evaluation_times.append(time()-start_tm)
                        hat_Pr[omega] = hat_Pr.get(omega,0) + 1.0/T
                        support_observed.append(omega)
                        if os.environ['precomp']:
                            save_pickle(omega, os.path.join(os.environ['precomp'],str(i)+"_"+str(j)+".pre"))
                            # j+=1
                # hat_H = -sum([hat_Pr[omega] * log2(hat_Pr[omega]) for omega in hat_Pr])
                hat_H = entropy([j for i,j in hat_Pr.items()], base = 2)
                hat_H_list.append(hat_H)
                sum_H += hat_H 
        else:
            for i in range(N):
                if self.Query.bucketing: #bucketing & pre
                    hat_Pr = {}
                    for j in range(T):
                        start_tm = time()
                        precomputed_omega_file = os.path.join(os.environ['precomp'],str(i)+"_"+str(j)+".pre")
                        omega = load_pickle(precomputed_omega_file)
                        query_evaluation_times.append(time()-start_tm)
                        # hat_Pr[omega] = hat_Pr.get(omega,0) + 1.0/T
                        support_observed.append(omega)
                    #bucketing 
                    # print(support_observed)
                    S = support_observed
                    _min,_max = min(S),max(S)
                    if _min == _max:
                        index_ub = 1
                        self.buckets = [(_min,_max)]
                    else:
                        delta = math.floor(math.sqrt(T))
                        index_ub = math.ceil((_max-_min)/delta)
                        self.buckets = [(_min+i*delta, _min+(i+1)*delta) for i in range(index_ub)]
                    # print('Buckets: ',self.buckets)
                    bucket_distr = {i: 0 for i in range(index_ub)}
                    for s in S:
                        omega_i = (1.0/T)
                        if _min < _max:
                            _index = math.floor((s - _min)/delta)
                            bucket_distr[_index] += omega_i
                        else:
                            bucket_distr[0] += omega_i
                    hat_Pr = bucket_distr
                else: #non-bucketing & pre
                    hat_Pr = {}
                    # for g in self.G.get_Ksample(T,seed=i):
                    for j in range(T):
                        start_tm = time()
                        precomputed_omega_file = os.path.join(os.environ['precomp'],str(i)+"_"+str(j)+".pre")
                        omega = load_pickle(precomputed_omega_file)
                        query_evaluation_times.append(time()-start_tm)
                        hat_Pr[omega] = hat_Pr.get(omega,0) + 1.0/T
                        support_observed.append(omega)
                hat_H = -sum([hat_Pr[omega] * log2(hat_Pr[omega]) for omega in hat_Pr])
                hat_H_list.append(hat_H)
                sum_H += hat_H 
        mean_H =  sum_H /N
        self.algostat['execution_time'] = time() - start_execution_time
        self.algostat['algorithm'] = 'appr'
        self.algostat['H'] = mean_H
        self.algostat['support'] =  str(support_observed)
        self.algostat['total_sample_tm'] = self.G.total_sample_tm
        self.algostat['query_eval_tm'] = sum(query_evaluation_times)
        return mean_H 

    def measure_uncertainty_rss(self, N=1, T=10, optimise = False):
        precomp = {}
        source,target = self.Query.u, self.Query.v
        if os.environ['precomp']:
            previous_omega_files = [f for f in os.listdir(os.environ['precomp']) if f.endswith('.pre')]
            # print(previous_omega_files)
            if len(previous_omega_files):
                for f in previous_omega_files:
                    # print('f = ',f)
                    n = f.split('.pre')[0]
                    n = int(n)
                    if n not in precomp:
                        precomp[n] = f

                maximum_N = max(precomp.keys())+1
                assert N <= maximum_N, "precomputed possible worlds are insufficient"

        start_execution_time = time()
        query_evaluation_times = []
        hat_H_list = []
        support_observed = []
        sum_H = 0
        if optimise:
            precomputed_nbrs_path = os.path.join('_'.join(os.environ['precomp'].split('_')[:-2])+"_nbr.pre")
        if len(precomp)==0: # -pre 0
            for i in range(N):
                hat_Pr = {}
                start_tm = time()
                if 'time_seed' in os.environ:
                    random.seed()
                    s = random.randint(0,1000000)
                else:
                    s = i
                if optimise and os.path.isfile(precomputed_nbrs_path):
                    # print('loading precomputed nbrs file..')
                    loaded_nbrs = load_pickle(precomputed_nbrs_path)
                    prOmega,nbrs = self.G.find_rel_rss(T,source,target,seed=s,optimiseargs = \
                                                    {'nbrs':loaded_nbrs,'doopt': True})  
                else:
                    prOmega,nbrs = self.G.find_rel_rss(T,source,target,seed=s,optimiseargs = None)                    
                query_evaluation_times.append(time()-start_tm)
                
                hat_Pr[1] =  prOmega
                hat_Pr[0] =  (1-prOmega)
                if os.environ['precomp']:
                    save_pickle(prOmega, os.path.join(os.environ['precomp'],str(i)+".pre"))
                    if not os.path.isfile(precomputed_nbrs_path):
                        # print('saving nbrs file for the 1st time.')
                        save_pickle(nbrs,precomputed_nbrs_path)
                # print(entropy([hat_Pr[1] ,hat_Pr[0]]))
                hat_H = entropy([hat_Pr[1] ,hat_Pr[0]])
                hat_H_list.append(hat_H)
                sum_H += hat_H 
        else: # -pre 1
            for i in range(N):
                hat_Pr = {}
                start_tm = time()
                precomputed_omega_file = os.path.join(os.environ['precomp'],precomp[i])
                prOmega = load_pickle(precomputed_omega_file)
                hat_Pr[1] =  prOmega
                hat_Pr[0] =  (1-prOmega)
                query_evaluation_times.append(time()-start_tm)
                hat_H = entropy([hat_Pr[1] ,hat_Pr[0]])
                hat_H_list.append(hat_H)
                sum_H += hat_H 
        mean_H =  sum_H /N
        self.algostat['execution_time'] = time() - start_execution_time
        self.algostat['algorithm'] = 'rss'
        self.algostat['H'] = mean_H
        self.algostat['support'] =  str(support_observed)
        self.algostat['total_sample_tm'] = self.G.total_sample_tm
        self.algostat['query_eval_tm'] = sum(query_evaluation_times)

    def measure_uncertainty_bfs(self, N=1, T=10, optimise = False, seed = 1):
        """
        MC + BFS 
        """
        precomp = {}
        source,target = self.Query.u, self.Query.v
        if os.environ['precomp']:
            previous_omega_files = [f for f in os.listdir(os.environ['precomp']) if f.endswith('.pre')]
            # print(previous_omega_files)
            if len(previous_omega_files):
                for f in previous_omega_files:
                    # print('f = ',f)
                    n,t = f.split('.pre')[0].split('_')
                    n,t = int(n),int(t)
                    if n not in precomp:
                        precomp[n] = {}
                    if t not in precomp[n]:
                        precomp[n][t] = f
                maximum_T = max(precomp[0].keys())+1
                maximum_N = max(precomp.keys())+1
                # print(maximum_N,maximum_T, N, T)
                assert N <= maximum_N, "precomputed possible worlds are insufficient"
                assert T <= maximum_T, "precomputed possible worlds are insufficient"
        start_execution_time = time()
        query_evaluation_times = []
        hat_H_list = []
        support_observed = []
        sum_H = 0
        if len(precomp)==0:
            for i in range(N):
                hat_Pr = {}
                j = 0
                if 'time_seed' in os.environ:
                    random.seed()
                    s = random.randint(0,1000000)
                else:
                    s = seed + i
                precomputed_nbrs_path = os.path.join('_'.join(os.environ['precomp'].split('_')[:-2])+"_nbr.pre")
                if optimise and os.path.isfile(precomputed_nbrs_path):
                    print('loading precomputed nbrs file..')
                    loaded_nbrs = load_pickle(precomputed_nbrs_path)
                    func_obj = self.G.get_Ksample_bfs(T,seed=s,source=source,target = target, optimiseargs = \
                                                      {'nbrs':loaded_nbrs,'doopt': True})
                else:
                    func_obj = self.G.get_Ksample_bfs(T,seed=s,source=source,target = target, optimiseargs = None)
                for nbrs,_, _,omega in func_obj:
                    start_tm = time()
                    # omega = self.Query.evalG(g[0])
                    query_evaluation_times.append(time()-start_tm)
                    hat_Pr[omega] = hat_Pr.get(omega,0) + 1.0/T
                    support_observed.append(omega)
                    if os.environ['precomp']:
                        save_pickle(omega, os.path.join(os.environ['precomp'],str(i)+"_"+str(j)+".pre"))
                        # precomputed_nbrs_path = os.path.join(os.environ['precomp']+"_nbr.pre")
                        if not os.path.isfile(precomputed_nbrs_path):
                            print('saving nbrs file for the 1st time.')
                            save_pickle(nbrs,precomputed_nbrs_path)
                        j+=1
                # hat_H = -sum([hat_Pr[omega] * log2(hat_Pr[omega]) for omega in hat_Pr])
                # print(hat_Pr.items())
                hat_H = entropy([j for i,j in hat_Pr.items()], base = 2)
                hat_H_list.append(hat_H)
                sum_H += hat_H 
        else:
            for i in range(N):
                hat_Pr = {}
                
                # for g in self.G.get_Ksample(T,seed=i):
                for j in range(T):
                    start_tm = time()
                    precomputed_omega_file = os.path.join(os.environ['precomp'],str(i)+"_"+str(j)+".pre")
                    omega = load_pickle(precomputed_omega_file)
                    query_evaluation_times.append(time()-start_tm)
                    hat_Pr[omega] = hat_Pr.get(omega,0) + 1.0/T
                    support_observed.append(omega)
                hat_H = -sum([hat_Pr[omega] * log2(hat_Pr[omega]) for omega in hat_Pr])
                hat_H_list.append(hat_H)
                sum_H += hat_H 
        mean_H =  sum_H /N
        self.algostat['execution_time'] = time() - start_execution_time
        self.algostat['algorithm'] = 'appr'
        self.algostat['H'] = mean_H
        self.algostat['support'] =  str(support_observed)
        self.algostat['total_sample_tm'] = self.G.total_sample_tm
        self.algostat['query_eval_tm'] = sum(query_evaluation_times)
        return mean_H 
    
    def measure_uncertainty_dhopbfs(self, d = 1, N=1, T=10, seed = 1):
        """
        MC + d-hop BFS 
        """
        # print("MC + d-hop BFS")
        source,target = self.Query.u, self.Query.v
        start_execution_time = time()
        query_evaluation_times = []
        hat_H_list = []
        support_observed = []
        sum_H = 0    
        for i in range(N):
            if 'time_seed' in os.environ:
                random.seed()
                s = random.randint(0,1000000)
            else:
                s = seed + i
            hat_Pr = {}
            func_obj = self.G.get_Ksample_dhopbfs(K = T,seed=s,\
                                        source=source,target = target, dhop = d, optimiseargs = None)
            for _,_, _,omega in func_obj:
                hat_Pr[omega] = hat_Pr.get(omega,0) + 1.0/T
                support_observed.append(omega)
            hat_H = -sum([hat_Pr[omega] * log2(hat_Pr[omega]) for omega in hat_Pr])
            hat_H_list.append(hat_H)
            sum_H += hat_H 
            # print(support_observed)
        mean_H =  sum_H /N
        self.algostat['execution_time'] = time() - start_execution_time
        self.algostat['algorithm'] = 'appr'
        self.algostat['H'] = mean_H
        self.algostat['support'] =  str(support_observed)
        self.algostat['total_sample_tm'] = self.G.total_sample_tm
        self.algostat['query_eval_tm'] = sum(query_evaluation_times)
        return mean_H 

    def measure_uncertainty_dij(self, N=1, T=10, optimise = False, seed = 1):
        """
        MC + Dijkstra
        """
        precomp = {}
        source,target = self.Query.u, self.Query.v
        if os.environ['precomp']:
            previous_omega_files = [f for f in os.listdir(os.environ['precomp']) if f.endswith('.pre')]
            # print(previous_omega_files)
            if len(previous_omega_files):
                for f in previous_omega_files:
                    # print('f = ',f)
                    n,t = f.split('.pre')[0].split('_')
                    n,t = int(n),int(t)
                    if n not in precomp:
                        precomp[n] = {}
                    if t not in precomp[n]:
                        precomp[n][t] = f
                maximum_T = max(precomp[0].keys())+1
                maximum_N = max(precomp.keys())+1
                # print(maximum_N,maximum_T, N, T)
                assert N <= maximum_N, "precomputed possible worlds are insufficient"
                assert T <= maximum_T, "precomputed possible worlds are insufficient"
        start_execution_time = time()
        query_evaluation_times = []
        hat_H_list = []
        support_observed = []
        sum_H = 0
        if len(precomp)==0:
            for i in range(N):
                hat_Pr = {}
                j = 0
                if 'time_seed' in os.environ:
                    random.seed()
                    s = random.randint(0,1000000)
                else:
                    s = seed + i
                precomputed_nbrs_path = os.path.join('_'.join(os.environ['precomp'].split('_')[:-2])+"_nbr.pre")
                if optimise and os.path.isfile(precomputed_nbrs_path):
                    # print('loading precomputed nbrs file..')
                    loaded_nbrs = load_pickle(precomputed_nbrs_path)
                    func_obj = self.G.get_Ksample_dij(T,seed=s,source=source,target = target, optimiseargs = \
                                                      {'nbrs':loaded_nbrs,'doopt': True})
                else:
                    func_obj = self.G.get_Ksample_dij(T,seed=s,source=source,target = target, optimiseargs = None)
                for nbrs,_, _,omega in func_obj:
                    start_tm = time()
                    # omega = self.Query.evalG(g[0])
                    query_evaluation_times.append(time()-start_tm)
                    hat_Pr[omega] = hat_Pr.get(omega,0) + 1.0/T
                    support_observed.append(omega)
                    if os.environ['precomp']:
                        if not os.path.isfile(precomputed_nbrs_path):
                            print('saving nbrs file for the 1st time.')
                            save_pickle(nbrs,precomputed_nbrs_path)
                        save_pickle(omega, os.path.join(os.environ['precomp'],str(i)+"_"+str(j)+".pre"))
                        j+=1
                hat_H = -sum([hat_Pr[omega] * log2(hat_Pr[omega]) for omega in hat_Pr])
                hat_H_list.append(hat_H)
                sum_H += hat_H 
        else:
            for i in range(N):
                hat_Pr = {}
                
                # for g in self.G.get_Ksample(T,seed=i):
                for j in range(T):
                    start_tm = time()
                    precomputed_omega_file = os.path.join(os.environ['precomp'],str(i)+"_"+str(j)+".pre")
                    omega = load_pickle(precomputed_omega_file)
                    query_evaluation_times.append(time()-start_tm)
                    hat_Pr[omega] = hat_Pr.get(omega,0) + 1.0/T
                    support_observed.append(omega)
                hat_H = -sum([hat_Pr[omega] * log2(hat_Pr[omega]) for omega in hat_Pr])
                hat_H_list.append(hat_H)
                sum_H += hat_H 
        mean_H =  sum_H /N
        self.algostat['execution_time'] = time() - start_execution_time
        self.algostat['algorithm'] = 'appr'
        self.algostat['H'] = mean_H
        self.algostat['support'] =  str(support_observed)
        self.algostat['total_sample_tm'] = self.G.total_sample_tm
        self.algostat['query_eval_tm'] = sum(query_evaluation_times)
        return mean_H 
    
    def measure_uncertainty_mctri(self, N=1, T=10, k= 265, optimise = False):
        precomp = {}
        if os.environ['precomp']:
            previous_omega_files = [f for f in os.listdir(os.environ['precomp']) if f.endswith('.pre')]
            # print(previous_omega_files)
            if len(previous_omega_files):
                for f in previous_omega_files:
                    # print('f = ',f)
                    n,t = f.split('.pre')[0].split('_')
                    n,t = int(n),int(t)
                    if n not in precomp:
                        precomp[n] = {}
                    if t not in precomp[n]:
                        precomp[n][t] = f
                maximum_T = max(precomp[0].keys())+1
                maximum_N = max(precomp.keys())+1
                # print(maximum_N,maximum_T, N, T)
                assert N <= maximum_N, "precomputed possible worlds are insufficient"
                assert T <= maximum_T, "precomputed possible worlds are insufficient"
        start_execution_time = time()
        query_evaluation_times = []
        hat_H_list = []
        support_observed = []
        sum_H = 0
        if len(precomp)==0:
            for i in range(N):
                if self.Query.bucketing: # buckeing & no-pre
                    if 'time_seed' in os.environ:
                        random.seed()
                        s = random.randint(0,1000000)
                    else:
                        s = i
                    hat_Pr = {}
                    j = 0
                    for g in self.G.get_Ksample(T,seed=s):
                        start_tm = time()
                        omega = self.G.count_tri_approx(g[0])
                        query_evaluation_times.append(time()-start_tm)
                        support_observed.append(omega)
                        if os.environ['precomp']:
                            save_pickle(omega, os.path.join(os.environ['precomp'],str(i)+"_"+str(j)+".pre"))
                            j+=1
                    #bucketing 
                    S = support_observed
                    _min,_max = min(S),max(S)
                    if _min == _max:
                        index_ub = 1
                        self.buckets = [(_min,_max)]
                    else:
                        delta = math.floor(math.sqrt(T))
                        index_ub = math.ceil((_max-_min)/delta)
                        self.buckets = [(_min+i*delta, _min+(i+1)*delta) for i in range(index_ub)]
                    # print(self.buckets)
                    bucket_distr = {x: 0 for x in range(index_ub)}
                    for s in S:
                        omega_i = (1.0/T)
                        if _min < _max:
                            _index = math.floor((s - _min)/delta)
                            bucket_distr[_index] += omega_i
                        else:
                            bucket_distr[0] += omega_i 
                    hat_Pr = bucket_distr
                    print([(self.buckets[_index],bucket_distr[_index]) for _index in bucket_distr])
                else: # non-bucketing & no-pre
                    if 'time_seed' in os.environ:
                        random.seed()
                        s = random.randint(0,1000000)
                    else:
                        s = i
                    hat_Pr = {}
                    for j in range(T):
                        start_tm = time()
                        g = self.G.get_sample(seed=s)
                        # omega = self.G.count_tri_approx(g[0])
                        # omega = self.G.count_tri_approx(g[0].nx_format.edges)
                        omega = self.G.count_tri_approx(g[0].Edges)
                        # omega = sum(nx.triangles(g[0].nx_format).values()) / 3
                        query_evaluation_times.append(time()-start_tm)
                        hat_Pr[omega] = hat_Pr.get(omega,0) + 1.0/T
                        support_observed.append(omega)
                        if os.environ['precomp']:
                            save_pickle(omega, os.path.join(os.environ['precomp'],str(i)+"_"+str(j)+".pre"))
                            j+=1
                    # print(i,' => ',hat_Pr)
                hat_H = entropy([j for i,j in hat_Pr.items()], base = 2)
                hat_H_list.append(hat_H)
                sum_H += hat_H 
        else:
            for i in range(N):
                if self.Query.bucketing: #bucketing & pre
                    hat_Pr = {}
                    for j in range(T):
                        start_tm = time()
                        precomputed_omega_file = os.path.join(os.environ['precomp'],str(i)+"_"+str(j)+".pre")
                        omega = load_pickle(precomputed_omega_file)
                        query_evaluation_times.append(time()-start_tm)
                        # hat_Pr[omega] = hat_Pr.get(omega,0) + 1.0/T
                        support_observed.append(omega)
                    #bucketing 
                    S = support_observed
                    _min,_max = min(S),max(S)
                    if _min == _max:
                        index_ub = 1
                        self.buckets = [(_min,_max)]
                    else:
                        delta = math.floor(math.sqrt(T))
                        index_ub = math.ceil((_max-_min)/delta)
                    self.buckets = [(_min+i*delta, _min+(i+1)*delta) for i in range(index_ub)]
                    # print(self.buckets)
                    bucket_distr = {x: 0 for x in range(index_ub)}
                    for s in S:
                        omega_i = (1.0/T)
                        if _min < _max:
                            _index = math.floor((s - _min)/delta)
                            bucket_distr[_index] += omega_i
                        else:
                            bucket_distr[0] += omega_i
                    hat_Pr = bucket_distr
                else: #non-bucketing & pre
                    hat_Pr = {}
                    # for g in self.G.get_Ksample(T,seed=i):
                    for j in range(T):
                        start_tm = time()
                        precomputed_omega_file = os.path.join(os.environ['precomp'],str(i)+"_"+str(j)+".pre")
                        omega = load_pickle(precomputed_omega_file)
                        query_evaluation_times.append(time()-start_tm)
                        hat_Pr[omega] = hat_Pr.get(omega,0) + 1.0/T
                        support_observed.append(omega)
                hat_H = -sum([hat_Pr[omega] * log2(hat_Pr[omega]) for omega in hat_Pr])
                hat_H_list.append(hat_H)
                sum_H += hat_H 
        mean_H =  sum_H /N
        self.algostat['execution_time'] = time() - start_execution_time
        self.algostat['algorithm'] = 'appr'
        self.algostat['H'] = mean_H
        self.algostat['support'] =  str(support_observed)
        self.algostat['total_sample_tm'] = self.G.total_sample_tm
        self.algostat['query_eval_tm'] = sum(query_evaluation_times)
        return mean_H 

    def algorithm3(self,  k, update_type = 'o1',verbose = False):
        """ Variant of Algorithm 3 where entropy is approximated via sampling, not exactly computed """
        # TODO: Sampler code: Ugraph().get_Ksample().
        self.algostat['algorithm'] = 'Alg3'
        self.algostat['k'] = k
        pass 

    def compute_Pr_up(self,verbose = False):
        """ 
        Calculate 
        Pr_up(G_i,e_j) (Pr_up(G_i) after p(e_j) is updated ) 
        """
        # assert op == 'o1'
        self.Pr_up = {}
        if (verbose):
            self.Gsums = {}
        for omega in self.Query.phiInv:
            self.Pr_up[omega] = {}
            if (verbose):
                self.Gsums[omega] = {}
            for e in self.Query.p_graph.edict:
                # p_e = self.G.edict[e] # p(e)
                _sumPrG = 0
                if (verbose):
                    self.Gsums[omega][e] = []
                for i in self.Query.phiInv[omega]:
                    if e in self.Query.index[i]: # if edge e is in G_i where omega is observed
                        p_e = self.Query.hatp[e] # using \hat{p}(e) instead of p(e) before update
                        # p_up_e = 1
                        # p_up_e = self.Query.p_graph.get_next_prob(e[0],e[1],op) # p_up(e)
                        G_up = self.Query.PrG[i]  / p_e  # Pr_up(G_i) = Pr(G) * p_up(e) / p(e)
                        _sumPrG += G_up 
                        if (verbose):
                            self.Gsums[omega][e].append((i,1))
                    # else: #  edge e not in G_i where omega is observed
                    #     # if p_e != p_up_e:
                    #     try:
                    #         G_up = self.Query.PrG[i] * (1-p_up_e) / (1-p_e)  # Pr_up(G_i) = Pr(G_i) * (1- p_up(e)) / (1- p(e))
                    #         _sumPrG += G_up 
                    #         if (verbose):
                    #             self.Gsums[omega][e].append((i,0))
                    #     except ZeroDivisionError:
                    #         print(p_e,p_up_e,self.Query.PrG[i])
                    #         raise ZeroDivisionError
                                
                self.Pr_up[omega][e] = _sumPrG
                        
    def measure_H0(self, property, algorithm, K, N = 1, seed = 1):
        if algorithm == 'exact':
            # print('exact')
            # self.Query.reset(self.G)
            self.Query.eval()
            H = self.Query.compute_entropy() # Initial entropy. Kept outside of time() because H0 is not needed in contr table and computed only for logging.
        elif algorithm == 'appr':
            H = self.measure_uncertainty(N=N, T = K, seed = seed)
        elif (algorithm == 'mcbfs' or algorithm == 'pTmcbfs'):
            print('seed = ',seed)
            if property == 'reach':
                assert (property == 'reach')
                H = self.measure_uncertainty_bfs(N=N, T = K, seed = seed)
            else:
                assert (property == 'reach_d')
                H = self.measure_uncertainty_dhopbfs(d = self.Query.d, N = N, T = K, seed = seed)
        elif algorithm == 'mcdij':
            assert (property == 'sp')
            H =  self.measure_uncertainty_dij(N=N, T = K,seed=seed)
        elif algorithm == 'mcapproxtri':
            assert (property == 'tri')
            H = self.measure_uncertainty_mctri(N=N, T = K)
        elif algorithm == 'rss':
            assert (property == 'reach')
            H = self.measure_uncertainty_rss(N=N, T = K)
        else:
            raise Exception('Undefined algorithm')
        if 'execution_time' in self.algostat:
            del self.algostat['execution_time']
        if 'algorithm' in self.algostat:
            del self.algostat['algorithm']
        if 'H' in self.algostat:
            del self.algostat['H']
        if 'support' in self.algostat:
            del self.algostat['support']
        if 'total_sample_tm' in self.algostat:
            del self.algostat['total_sample_tm']
        if 'query_eval_tm' in self.algostat:
            del self.algostat['query_eval_tm'] 
        return H 
    
    def algorithm5(self, property, algorithm, k, K, N=1,T=1, update_type = 'o1', verbose = False, track_H = False):
        """ Variant of Algorithm 5 where CT is approximated via sampling"""
        assert k>=1
        history = []
        start_execution_time = time()
        if verbose: print('Query eval.')
        self.algostat['result']['H0'] = self.measure_H0(property,algorithm,T,N)
        H0 = self.algostat['result']['H0']
        # Start of Algorithm 5
        self.Query.constructTables_S(K = K, verbose = verbose) 
        Estar = copy(self.G.edict)
        assert k<=len(Estar)
        E = []
        if (verbose):
            print('H0: ',self.algostat['result']['H0'])
            # print('p(e): ',self.G.edict)
            print('\hatp(e): ',self.Query.hatp)
            print('Pr[G]: ',self.Query.PrG)
            # print('G: ',self.Query.G)
            # print('Pr[Omega]: ', self.Query.freq_distr)
            # print('results: ',self.Query.results)
        # Calculate Pr_up^{e_j}(G_i) for all i,j
        self.compute_Pr_up(verbose = verbose)
        can_reduce_further = True
        history_deltaH = []
        history_Hk = []
        previous_pe = []
        for iter in range(k):
            if verbose: print('------ iter = ',iter,' ------')
            if not can_reduce_further:
                if verbose: print('R = 0. stop')
                break 
            if (verbose):
                print("Iteration: ",iter)
            # Start of Algorithm 4
            # print('Pr(G_i): ', self.Query.PrG)
            # Construct Pr[Omega] 
            Pr_Omega = {}
            for omega in self.Query.phiInv:
                Pr_Omega[omega] = sum([self.Query.PrG[i] for i in self.Query.phiInv[omega]])
            H0_i = entropy([j for i,j in Pr_Omega.items()], base = 2)
            if self.debug: print('Pr[omega]: ',Pr_Omega)
            if self.debug:  print("M before update: \n\r",self.Pr_up)
            if self.debug:  M = deepcopy(self.Pr_up)#; print(M)
            if (verbose):
                # self.Query.eval()
                # H0 = self.measure_H0(property,algorithm,T,N)
                print('H_0(before e*_k) = ', entropy([j for i,j in Pr_Omega.items()], base = 2))

            if (verbose):
                print('Selecting edge#: ',iter+1)
                # print('Jacobian: ', DelPr_Omega_pe)
                print('Pr[Omega]: ',Pr_Omega)
                print("Pr_up[Omega]: ",self.Pr_up)
                # print('Gsums: ',self.Gsums)
            # local_maxima = None 
            minima = []
            candidates = []
            for e in Estar: # Among remaining edges (Estar), find the one (e*) with largest  H - H_{up}
                DeltaHe2 = 0 # H - H^{e}_{up}
                H_up = 0
                # for omega in self.Query.phiInv:
                #     if (Pr_Omega[omega] != 0):
                #         if (self.Pr_up[omega].get(e,0) != 0):
                #             # print(e,omega,' =>',self.Pr_up[omega][e])
                #             # print(Pr_Omega[omega])
                #             DeltaHe2 += (log2(self.Pr_up[omega][e])*self.Pr_up[omega][e] - log2(Pr_Omega[omega])*Pr_Omega[omega])
                #             H_up += (-log2(self.Pr_up[omega][e])*self.Pr_up[omega][e])
                #         else:
                #             DeltaHe2 += (0 - log2(Pr_Omega[omega])*Pr_Omega[omega])
                #     else:
                #         if (self.Pr_up[omega].get(e,0) != 0):
                #             DeltaHe2 += (log2(self.Pr_up[omega][e])*self.Pr_up[omega][e] - 0)
                #             H_up += (-log2(self.Pr_up[omega][e])*self.Pr_up[omega][e])
                #         else:
                #             DeltaHe2 += 0
                # # if verbose:
                # if self.debug:
                #     print('H_up^e', e,' -> ',H_up)
                # if verbose:
                #     print('e: ',e,' DeltaH2[e]: ',DeltaHe2,' \hat{H}_'+str(iter+1),' = ',H_up)
                #     # print('H_next [',e,']: ', self.algostat['result']['H0'] + DeltaHe2)
                
                # if local_maxima is not None:
                #     if DeltaHe2 >= local_maxima[1]:
                #         local_maxima = (e,DeltaHe2)
                #         if verbose:
                #             print('new maxima: ',local_maxima)
                # else:
                #    local_maxima = (e,DeltaHe2)

                # if local_maxima is not None:
                #     if H0 -  H_up > H0 - local_maxima[1]:
                #         local_maxima = (e, H0-H_up)
                #         if (verbose):
                #             print('new maxima: ',local_maxima)
                # else:
                #     local_maxima = (e, H0-H_up)
                for omega in self.Query.phiInv:
                    H_up += h(self.Pr_up[omega].get(e,0))
                minima.append(H_up)
                candidates.append(e)
            minima_index = np.argmin(minima)
            if verbose:
                top_k = np.argsort(minima)
                print('Round ',iter,' : ')
                for i,j in zip(np.array(minima)[top_k],np.array(candidates)[top_k]):
                    print(j,' => ',i)
                print('Selected edge: ',candidates[minima_index])
            if verbose: print(minima_index," : ",minima[minima_index])
            multiple_minima = np.where(minima[minima_index] == minima)[0]
            if verbose and len(multiple_minima)>1: 
                print('multiple candidates: ', candidates[multiple_minima])
            # print(len(Estar),' ',len(minima))
            local_maxima = (candidates[minima_index],H0_i - minima[minima_index])
            if verbose: print('e*: ',local_maxima[0],' DeltaH2[e]: ',local_maxima[1])
            estar = local_maxima[0] # assign e*
            history_deltaH.append(local_maxima[1])
            # history_Hk.append(H_up)
            history_Hk.append(minima[minima_index])
            previous_pe.append(self.G.get_prob(estar))
            # H0 = H_up
            # End of Algorithm 4
            E.append(estar)
            del Estar[estar] # Delete e* from dictionary s.t. next iteration is 1 less than the current.
            if iter < k-1:
                can_reduce_further = self.Query.updateTables(estar, self.Pr_up)
            else:
                if self.debug:
                    self.Query.updateTables(estar, self.Pr_up)
                    # self.G.edge_update(estar[0], estar[1], type = update_type)
                    # self.Query.reset(self.G)
                    # self.compute_Pr_up(update_type, verbose = verbose)
                    # print('-----')
                    print('M after update: ',self.Pr_up)

            self.G.edge_update(estar[0],estar[1],type = update_type) # Update UGraph()
            # print('-',self.Query.PrG)
            self.Query.reset(self.G)    # Re-initialise Query() with updated UGraph()  
            if track_H:
                history.append(self.algostat['result']['H0']-self.measure_H0(property,algorithm,T,N)) 
            # print('=',self.Query.PrG)         
            if (verbose):
                print('After p(e) update: ')
                print('Pr(G_i): ', self.Query.PrG)
                print('\hatp(e): ',self.Query.hatp)
                print('history[DeltaH] = ',history_deltaH)
                # print('C = ',self.Query.C)
        history_Hk = np.array(history_Hk)
        if verbose: print('history_Hk = ',history_Hk)
        h0_hkhat = self.algostat['result']['H0'] - history_Hk
        min_hkhat = np.argmin(history_Hk)
        minima_index_list = np.where(history_Hk[min_hkhat] == history_Hk)[0]
        if verbose and len(minima_index_list)>1: print('>1 minima: ', minima_index_list,' ',history_Hk[minima_index_list])
        largest_minima_index = minima_index_list[-1]
        if verbose: print('argmin \hat{H}_k= ',min_hkhat, '. \hat{H}_k* = ',history_Hk[largest_minima_index])
        
        for i in range(largest_minima_index+1,len(E)):
            self.Query.p_graph.update_edge_prob(E[i][0],E[i][1],previous_pe[i])
        E = E[:largest_minima_index+1] # Take edges until the argmax min(H_1,H_2,..,H_k)
        if verbose: print('H0-\hat{H}k: ',h0_hkhat)
        e_tm = time() - start_execution_time
        self.algostat['algorithm'] = 'greedy+mem'
        self.algostat['MCalgo'] = algorithm
        self.algostat['k'] = k
        self.algostat['result']['edges'] = E
        # self.Query.eval()
        # self.algostat['result']['H*'] = self.algostat['result']['H0']-history[-1]
        self.algostat['result']['H*'] = self.measure_H0(property,algorithm,T,N)
        self.algostat['execution_time'] = e_tm
        # self.algostat['support'] = str(list(self.Query.phiInv.keys()))
        self.algostat['DeltaH'] = self.algostat['result']['H0'] - self.algostat['result']['H*']
        self.algostat['|DeltaH|'] = abs(self.algostat['result']['H0'] - self.algostat['result']['H*'])
        self.algostat['history_deltaH'] = history
        if self.debug: self.algostat['M'] = M
        return E,self.algostat['result']['H*'], self.algostat['result']['H0']- self.algostat['result']['H*']
    
    # def algorithm6(self, property, algorithm, k, K, update_type = 'o1',verbose = False):
    #     """ Variant of Algorithm 5 where CT is approximated via sampling"""
    #     if verbose: print('Query eval.')
    #     self.algostat['result']['H0'] = self.measure_H0(property,algorithm,K)
    #     start_execution_time = time()
    #     # Start of Algorithm 5
    #     self.Query.constructTables_S(K = K, verbose = verbose) 
    #     Estar = copy(self.G.edict)
    #     E = []
    #     if (verbose):
    #         print('H0: ',self.algostat['result']['H0'])
    #         print('p(e): ',self.G.edict)
    #         print('Pr[G]: ',self.Query.PrG)
    #         print('G: ',self.Query.G)
    #         # print('Pr[Omega]: ', self.Query.freq_distr)
    #         # print('results: ',self.Query.results)
    #     for iter in range(k):
    #         if (verbose):
    #             print("Iteration: ",iter)
    #             print('Pr[G]: ',self.Query.PrG)
                
    #         # Start of Algorithm 4
            
    #         # Construct Pr[Omega] 
    #         Pr_Omega = {}
    #         for omega in self.Query.phiInv:
    #             Pr_Omega[omega] = sum([self.Query.PrG[i] for i in self.Query.phiInv[omega]])

    #         # Calculate Pr_up^{e_j}(G_i) for all i,j
    #         self.compute_Pr_up(verbose = verbose)
            
    #         if (verbose):
    #             # self.Query.eval()
    #             H0 = self.measure_H0(property,algorithm,K)
    #             # print('H0: ',H0)

    #         if (verbose):
    #             print('Selecting edge#: ',iter+1)
    #             print('Pr[Omega]: ',Pr_Omega)
    #             print("Pr_up[Omega]: ",self.Pr_up)
    #             # print('Gsums: ',self.Gsums)
    #         local_maxima = None 
    #         for e in Estar: # Among remaining edges (Estar), find the one (e*) with largest  H - H_{up}
    #             DeltaHe2 = 0 # H - H^{e}_{up}
    #             for omega in self.Query.phiInv:
    #                 if (Pr_Omega[omega] != 0):
    #                     if (self.Pr_up[omega][e] != 0):
    #                         DeltaHe2 += (log2(self.Pr_up[omega][e])*self.Pr_up[omega][e] - log2(Pr_Omega[omega])*Pr_Omega[omega])
    #                     else:
    #                         DeltaHe2 += (0 - log2(Pr_Omega[omega])*Pr_Omega[omega])
    #                 else:
    #                     if (self.Pr_up[omega][e] != 0):
    #                         DeltaHe2 += (log2(self.Pr_up[omega][e])*self.Pr_up[omega][e] - 0)
    #                     else:
    #                         DeltaHe2 += 0

    #             if verbose:
    #                 print('e: ',e,' DeltaH2[e]: ',DeltaHe2)
    #                 print('H_next [',e,']: ', H0 + DeltaHe2)
                
    #             if local_maxima is not None:
    #                 if DeltaHe2 > local_maxima[1]:
    #                     local_maxima = (e,DeltaHe2)
    #             else:
    #                local_maxima = (e,DeltaHe2)
        
    #         estar = local_maxima[0] # assign e*
    #         print('e: ',local_maxima[0],' DeltaH2[e]: ',local_maxima[1])
    #         # End of Algorithm 4
    #         E.append(estar)
    #         if (verbose):
    #             print('e* = ',estar)
    #         del Estar[estar] # Delete e* from dictionary s.t. next iteration is 1 less than the current.
    #         self.G.edge_update(estar[0],estar[1],type = update_type) # Update UGraph()
    #         self.Query.reset(self.G) # Re-initialise Query() with updated UGraph()
    #         # Instead of updateing the table we construct from scratch by re-sampling
    #         if iter < k-1:
    #             self.Query.constructTables_S(K = K, verbose = verbose) 

    #         if (verbose):
    #             print('After p(e) update: ')
    #             # print('C = ',self.Query.C)
    #         if (verbose):
    #             print('----------')
    #     self.algostat['algorithm'] = 'greedy+mem+resamp'
    #     self.algostat['MCalgo'] = algorithm
    #     self.algostat['k'] = k
    #     self.algostat['result']['edges'] = E
    #     self.algostat['result']['H*'] = self.measure_H0(property,algorithm,K)
    #     self.algostat['execution_time'] = time() - start_execution_time
    #     # self.algostat['support'] = str(list(self.Query.phiInv.keys()))
    #     self.algostat['DeltaH'] = self.algostat['result']['H0'] - self.algostat['result']['H*']
    #     self.algostat['|DeltaH|'] = abs(self.algostat['result']['H0'] - self.algostat['result']['H*'])
    #     return E,self.algostat['result']['H*'],self.algostat['result']['H0'] - self.algostat['result']['H*']
    
    #     # Greedy k-edge selection: Algorithm 3
    def compute_influence_set(self, nodes, E, nx_G, d):
        G = nx_G
        influence_set = {}
        for e in E:
            u,v = min(e[0],e[1]),max(e[0],e[1])
            length_v_all = nx.single_source_shortest_path_length(G, v,cutoff=d)
            length_all_u = dict(nx.single_target_shortest_path_length(G, u,cutoff=d))
            for s,t in combinations(nodes,2):
                _s,_t = min(s,t), max(s,t)
                dv_t = length_v_all.get(_t,math.inf)
                dv_s = length_v_all.get(_s,math.inf)
                dt_u = length_all_u.get(_t,math.inf)
                ds_u = length_all_u.get(_s,math.inf)
                uv_weight = self.G.weights[(u,v)]
                if (ds_u+uv_weight+dv_t<=d) or (dv_s + uv_weight+dt_u <= d):
                    influence_set[(u,v)] = influence_set.get((u,v),[])
                    influence_set[(u,v)].append((_s,_t))
        return influence_set
    def greedy(self,  property, algorithm, k,  N = 1, T = 1, update_type = 'o1', verbose = False, track_H = False):
        """
        Exact version of Algorithm 3
        Returns selected edgeset, entropy value after selection, and entropy-reduction amount (DeltaH)
        """
        assert k>=1
        history = []
        self.algostat['algorithm'] = 'greedy'
        self.algostat['k'] = k
        start_execution_time = time()
        Estar = copy(self.G.edict)
        assert k<=len(Estar)
        # Ecand = [('s','u'),('s','y')]
        # Ecand = [('s','x'),('s','y')]
        # self.Query.eval()
        # H0 = self.Query.compute_entropy() # Initial entropy
        init_seed = int(str(self.Query.u)+str(self.Query.v))
        H0 = self.measure_H0(property,algorithm,T, N)
        self.algostat['result']['H0'] = H0
        E = []
        history_Hk = [H0]
        previous_pe = [0]
        if (verbose):
            print('H0: ',H0)
            print('p(e): ',self.G.edict)
            print('Pr[Omega]: ', self.Query.freq_distr)
            # print('results: ',self.Query.results)
        for i in range(k):
            # print('iteration: ',i)
            # local_maxima = None 
            if (verbose):
                print('Selecting edge#: ',i+1)
            # Estar= [Ecand[i]]
            minima = []
            candidates = []
            for e in Estar: # Among remaining edges (Estar), find the one (e*) with largest reduction in Entropy
                g_copy = deepcopy(self.G)
                # if (verbose):
                #     print('g_copy: ',g_copy.edict)
                g_copy.edge_update(e[0],e[1], type= update_type)
                # if i==0:
                #     g_copy.update_edge_prob(e[0],e[1], 0)
                # else:
                #     g_copy.update_edge_prob(e[0],e[1], 1)
                self.Query.reset(g_copy)
                # self.Query.eval()
                # if (verbose):
                #     print('considering e= ',e)
                #     # print('p(e): ',g_copy.edict)
                #     print('Pr[Omega]: ', self.Query.freq_distr)
                #     # print('results: ',self.Query.results)
                # Hi = self.Query.compute_entropy()
                H_up = self.measure_H0(property, algorithm, T, N,seed=init_seed+i)
                minima.append(H_up)
                candidates.append(e)
                # if local_maxima:
                #     if H0 -  H_up > H0 - local_maxima[1]:
                #         local_maxima = (e, H_up)
                #         # if (verbose):
                #         #     print('new maxima: ',local_maxima)
                # else:
                #     local_maxima = (e, H_up)
                    # if (verbose):
                    #     print('initial maxima: ',local_maxima)
                # print('e: ',e,' \Delta H = ', H0-H_up, ' H0:', H0, ' H_next: ',H_up)
            # if verbose: print('H_up : ',minima,'\n\r candidates: ',candidates)
            if verbose:
                print('Round ',i,' : ')
                for x,y in zip(minima,candidates):
                    print(y,' => ',x)
            minima_index = np.argmin(minima)
            multiple_minima = np.where(minima[minima_index] == minima)[0]
            local_maxima = (candidates[minima_index],H0 - minima[minima_index])
            estar = local_maxima[0] # assign e*
            H0 = local_maxima[1] # Assign H0 for the next iteration.
            history_Hk.append(minima[minima_index])
            previous_pe.append(self.G.get_prob(estar))
            E.append(estar)
            if (verbose):
                print('e* = ',estar)
            del Estar[estar] # Delete e* from dictionary s.t. next iteration is 1 less than the current.
            self.G.edge_update(estar[0],estar[1],type = update_type) # Update UGraph()
            self.Query.reset(self.G) # Re-initialise Query() with updated UGraph()
            if track_H:
                history.append(self.algostat['result']['H0']-self.measure_H0(property,algorithm,T,N))
            # if (verbose):
            #     self.Query.eval()
            #     print('Entropy of Updated UGraph: ',self.Query.compute_entropy())
        history_Hk = np.array(history_Hk)
        if verbose: print('history_Hk = ',history_Hk)
        h0_hkhat = self.algostat['result']['H0'] - history_Hk # H0-\hat{H}_k
        
        min_hkhat = np.argmin(history_Hk)
        # minima_index_list = np.where(history_Hk[min_hkhat] == history_Hk)[0]
        # if verbose and len(minima_index_list)>1: print('>1 minima: ', minima_index_list,' ',history_Hk[minima_index_list])
        # largest_minima_index = minima_index_list[-1]
        # if verbose: print('argmin \hat{H}_k= ',min_hkhat, '. \hat{H}_k* = ',history_Hk[largest_minima_index])
        
        # # for i in range(largest_minima_index+1,len(E)):
        # #     self.Query.p_graph.update_edge_prob(E[i][0],E[i][1],previous_pe[i])
        # E = E[:largest_minima_index+1] # Take edges until the argmax min(H_1,H_2,..,H_k)
        # if verbose: print('H0-\hat{H}k: ',h0_hkhat)
        
        e_tm = time() - start_execution_time
        self.algostat['MCalgo'] = algorithm
        self.algostat['result']['edges'] = E[:min_hkhat]
        self.algostat['result']['H*'] =  history_Hk[min_hkhat] #self.measure_H0(property,algorithm,T,N)
        self.algostat['execution_time']= e_tm
        self.algostat['support'] = ''
        self.algostat['DeltaH'] = self.algostat['result']['H0'] - self.algostat['result']['H*']
        self.algostat['|DeltaH|'] = abs(self.algostat['result']['H0'] - self.algostat['result']['H*'])
        self.algostat['history_Hk'] = ",".join(["{:.4f}".format(i) for i in history_Hk])
        return E,self.algostat['result']['H*'],self.algostat['result']['H0']- self.algostat['result']['H*']

    def dp(self,  property, algorithm, k,  N = 1, T = 1, update_type = 'o1', verbose = False, track_H = False):
        """
        Bruteforce algorithm that shows the search Tree of edges.
        """
        ZERO = 10**(-13)
        def h(x):
            absx = abs(x)
            if absx <= ZERO:
                return 0
            elif (1-absx) <= ZERO:
                return 0
            else:
                try:
                    p = log2(x)
                except:
                    print(x)
            return -x*log2(x)
        
        self.algostat['algorithm'] = 'greedy+dp'
        self.algostat['k'] = k
        start_execution_time = time()
        H0 = self.measure_H0(property,algorithm,T, N)
        self.algostat['result']['H0'] = H0
        E = []
        Estar = copy(self.G.edict)
        memory = {}
        head = Node('[], H0 = '+"{:1.3f}".format(H0))
        for i in range(1,k+1):
            print('--> ',i)
            # for choice in permutations(Estar.keys(),i):
            for choice in combinations(Estar.keys(),i):
                print(choice)
                if len(choice)==1:
                    g_copy = deepcopy(self.G)
                    g_copy.edge_update(choice[0][0],choice[0][1], type= update_type)
                    self.Query.reset(g_copy)
                    H_up = self.measure_H0(property, algorithm, T, N)
                    if choice not in memory:
                        memory[choice] = Node(str(choice)+'='+"{:1.3f}".format(H_up) \
                                              +'/' +"{:1.3f}".format(h(self.G.get_prob(choice[0]))), head)

                else:
                    g_copy = deepcopy(self.G)
                    H_pe = 0
                    for e in choice:
                        H_pe += h(self.G.get_prob(e))
                        g_copy.edge_update(e[0],e[1], type= update_type)
                    self.Query.reset(g_copy)
                    H_up = self.measure_H0(property, algorithm, T, N)
                    ref = memory[tuple(choice[:-1])]
                    # for j,e in enumerate(choice):
                    #     if e in memory:
                    #         ref = memory[e]
                    #         continue 
                    #     else:
                    memory[choice] = Node(str(choice)+'='+"{:1.3f}".format(H_up)+'/'+"{:1.3f}".format(H_pe), ref)
        print_tree(head)
        # print_tree(head,horizontal = False)
        # json_object = json.dumps(memory,indent = 4)
        # json_formatted_str = json.dumps(json_object, indent=2)
        # print(json_formatted_str)
        # print(json_object)
        # assert k>=1
        # history = []
        # self.algostat['algorithm'] = 'greedy'
        # self.algostat['k'] = k
        # if(len(self.G.nx_format) == 0):
        #     self.G.nx_format.add_edges_from(self.G.Edges)
        # nodes = self.G.nx_format.nodes()
        # nx_G = self.G.nx_format
        # start_execution_time = time()
        # Estar = copy(self.G.edict)
        # # if Iset:
        # #     inf_set = self.compute_influence_set(nodes,self.G.edict.keys(),nx_G,self.Query.d)
        # #     # Estar = inf_set
        # #     print('influence_set: ',inf_set)

        # assert k<=len(Estar)
        # # Ecand = [('s','u'),('s','y')]
        # # Ecand = [('s','x'),('s','y')]
        # # self.Query.eval()
        # # H0 = self.Query.compute_entropy() # Initial entropy
        # H0 = self.measure_H0(property,algorithm,T, N)
        
        # self.algostat['result']['H0'] = H0
        # E = []
        # history_Hk = []
        # previous_pe = []
        # if (verbose):
        #     print('H0: ',H0)
        #     print('p(e): ',self.G.edict)
        #     print('Pr[Omega]: ', self.Query.freq_distr)
        #     # print('results: ',self.Query.results)
        # for i in range(k):
        #     print('iteration: ',i)
        #     # local_maxima = None 
        #     if (verbose):
        #         print('Selecting edge#: ',i+1)
        #     # Estar= [Ecand[i]]
        #     minima = []
        #     candidates = []
        #     # if Iset:
        #     #     qual_improve = {}
        #     #     prev_qual = -1000000000
        #     #     for e in Estar:
        #     #         g_copy = deepcopy(self.G)
        #     #         g_copy.edge_update(e[0],e[1], type= update_type)
        #     #         self.Query.reset(g_copy)
        #     #         He = self.measure_H0(property, algorithm, T, N)
        #     #         qual_improve[e] = H0 - He
        #     #         candidates.append(e)
        #     #         minima.append(qual_improve[e])
        #     #     if verbose:
        #     #         print('qual delta: ', qual_improve)
        #     # for e in Estar: # Among remaining edges (Estar), find the one (e*) with largest reduction in Entropy
        #     #     print('e = ',e)
        #     #     g_copy = deepcopy(self.G)
        #     #     # if (verbose):
        #     #     #     print('g_copy: ',g_copy.edict)
        #     #     g_copy.edge_update(e[0],e[1], type= update_type)
        #     #     # if i==0:
        #     #     #     g_copy.update_edge_prob(e[0],e[1], 0)
        #     #     # else:
        #     #     #     g_copy.update_edge_prob(e[0],e[1], 1)
        #     #     self.Query.reset(g_copy)
        #     #     # self.Query.eval()
        #     #     # if (verbose):
        #     #     #     print('considering e= ',e)
        #     #     #     # print('p(e): ',g_copy.edict)
        #     #     #     print('Pr[Omega]: ', self.Query.freq_distr)
        #     #     #     # print('results: ',self.Query.results)
        #     #     # Hi = self.Query.compute_entropy()
        #     #     if Iset:
        #     #         Qual_next = 0
        #     #         for _e in inf_set[e]:
        #     #             Qual_next += qual_improve.get(_e,0)
        #     #             Qual_next += qual_improve.get((_e[1],_e[0]),0)
        #     #         minima.append(Qual_next)
        #     #     else:
        #     #         H_up = self.measure_H0(property, algorithm, T, N)
        #     #         minima.append(H_up)
        #     #     candidates.append(e)
        #         # if local_maxima:
        #         #     if H0 -  H_up > H0 - local_maxima[1]:
        #         #         local_maxima = (e, H_up)
        #         #         # if (verbose):
        #         #         #     print('new maxima: ',local_maxima)
        #         # else:
        #         #     local_maxima = (e, H_up)
        #             # if (verbose):
        #             #     print('initial maxima: ',local_maxima)
        #         # print('e: ',e,' \Delta H = ', H0-H_up, ' H0:', H0, ' H_next: ',H_up)
        #     # if verbose: print('H_up : ',minima,'\n\r candidates: ',candidates)
        #     if verbose:
        #         print('Round ',i,' : ')
        #         for x,y in zip(minima,candidates):
        #             print(y,' => ',x)
        #     # if Iset:
        #     #     maxima_index = np.argmax(minima)
        #     #     local_maxima =  (candidates[maxima_index],H0 - minima[maxima_index])
        #     #     estar = local_maxima[0]
        #     #     Hupstar = H0 - qual_improve[estar]
        #     #     history_Hk.append(Hupstar)
        #     #     H0 = Hupstar
        #     else:
        #         minima_index = np.argmin(minima)
        #         multiple_minima = np.where(minima[minima_index] == minima)[0]
        #         local_maxima = (candidates[minima_index],H0 - minima[minima_index])
        #         estar = local_maxima[0] # assign e*
        #         H0 = local_maxima[1] # Assign H0 for the next iteration.
        #         history_Hk.append(minima[minima_index])
        #     previous_pe.append(self.G.edict[estar])
        #     E.append(estar)
        #     del inf_set[estar]
        #     if (verbose):
        #         print('e* = ',estar)
        #     del Estar[estar] # Delete e* from dictionary s.t. next iteration is 1 less than the current.
        #     self.G.edge_update(estar[0],estar[1],type = update_type) # Update UGraph()
        #     self.Query.reset(self.G) # Re-initialise Query() with updated UGraph()
        #     if track_H:
        #         history.append(self.algostat['result']['H0']-self.measure_H0(property,algorithm,T,N))
        #     # if (verbose):
        #     #     self.Query.eval()
        #     #     print('Entropy of Updated UGraph: ',self.Query.compute_entropy())
        # history_Hk = np.array(history_Hk)
        # if verbose: print('history_Hk = ',history_Hk)
        # h0_hkhat = self.algostat['result']['H0'] - history_Hk # H0-\hat{H}_k
        
        # min_hkhat = np.argmin(history_Hk)
        # minima_index_list = np.where(history_Hk[min_hkhat] == history_Hk)[0]
        # if verbose and len(minima_index_list)>1: print('>1 minima: ', minima_index_list,' ',history_Hk[minima_index_list])
        # largest_minima_index = minima_index_list[-1]
        # if verbose: print('argmin \hat{H}_k= ',min_hkhat, '. \hat{H}_k* = ',history_Hk[largest_minima_index])
        
        # for i in range(largest_minima_index+1,len(E)):
        #     self.Query.p_graph.update_edge_prob(E[i][0],E[i][1],previous_pe[i])
        # E = E[:largest_minima_index+1] # Take edges until the argmax min(H_1,H_2,..,H_k)
        # if verbose: print('H0-\hat{H}k: ',h0_hkhat)
        
        # e_tm = time() - start_execution_time
        # self.algostat['MCalgo'] = algorithm
        # self.algostat['result']['edges'] = E
        # self.algostat['result']['H*'] = self.measure_H0(property,algorithm,T,N)
        # self.algostat['execution_time']= e_tm
        # self.algostat['support'] = ''
        # self.algostat['DeltaH'] = self.algostat['result']['H0'] - self.algostat['result']['H*']
        # self.algostat['|DeltaH|'] = abs(self.algostat['result']['H0'] - self.algostat['result']['H*'])
        # self.algostat['history_deltaH'] = history
        # return E,self.algostat['result']['H*'],self.algostat['result']['H0']- self.algostat['result']['H*']

    def greedyP(self,  property, algorithm, k, r, N = 1, T = 1, update_type = 'o1', verbose = False, track_H = False):
        """
        Bruteforce algorithm that shows the search Tree of edges.
        """
        assert update_type=='o1'
        ZERO = 10**(-13)
        def h(x):
            absx = abs(x)
            if absx <= ZERO:
                return 0
            elif (1-absx) <= ZERO:
                return 0
            else:
                try:
                    p = log2(x)
                except:
                    print(x)
            return -x*log2(x)
        
        def weightFn(e, type = 'log'): # type = `log` (-p(e)logp(e)), `hx` (H(e)), None (1), orig (w(e))
            if type is None:
                return 1
            else:
                x = self.G.get_prob(e)
                if type == 'log':
                    return h(x) # -p(e)log p(e)
                elif type == 'hx':
                    return h(x)+h(1-x) # Entropy(e)
                elif type == 'orig':
                    return self.G.weights[e]
        self.algostat['algorithm'] = 'greedy+struct'
        self.algostat['k'] = k
        if property!='tri':
            init_seed = int(str(self.Query.u)+str(self.Query.v))
        H0 = self.measure_H0(property,algorithm,T, N)
        self.algostat['result']['H0'] = H0

        start_execution_time = time()
        E = []
        history_Hk = [H0]
        top_rpaths = []
        edge_path_index = {}
        if property!='tri':
            nx_G = self.G.nx_format
            # edges_with_weights = [(e[0],e[1],weightFn(e,type=None)) for e in self.G.Edges]
            # nx_G.add_weighted_edges_from(edges_with_weights)
            nx_G.add_edges_from(self.G.Edges)
            if verbose: print(nx_G.edges(data=True))
            toppath = nx.shortest_path(nx_G,self.Query.u,self.Query.v)
            toppath_len = len(toppath)
            if verbose: print(toppath)
            structure_len = [0]
            if toppath_len-1 <= k: # If path length within budget select all edges
                # E = [(toppath[j],toppath[j+1]) for j in range(toppath_len-1)]
                for j in range(toppath_len-1):
                    u,v = toppath[j],toppath[j+1]
                    E.append((u,v))
                    self.G.edge_update(u,v, type= update_type)
                    self.Query.reset(self.G)
                structure_len+=[i for i in range(1,toppath_len)]
                # print('adding all edges in the top-path: ',E)
                H_up = self.measure_H0(property, algorithm, T, N, seed = init_seed+k-1)
                # print('after cleaning ',E[-3:],' H_up = ',H_up)
                history_Hk.append(H_up)
            else: # If path length exceeds budget select only k-best entropy edges.
                t = 'hx'
                edges_in_top_path = [(toppath[j],toppath[j+1],weightFn((toppath[j],toppath[j+1]),t)) \
                                    for j in range(toppath_len-1)]
                edges_in_top_path.sort(key = lambda x: -x[-1])
                for count in range(k):
                    u,v,edge_entropy = edges_in_top_path[count]
                    # print(u,v,' => ',edge_entropy)
                    E.append((u,v))
                    self.G.edge_update(u,v, type= update_type)
                    self.Query.reset(self.G)
                    H_up = self.measure_H0(property, algorithm, T, N, seed = init_seed+count)
                    # print('after cleaning ',E[-3:],' H_up = ',H_up)
                    history_Hk.append(H_up)
                structure_len+=[i for i in range(1,k+1)]

        elif property == 'tri':
            method = "opt1" #"apprx"
            weight_type = 'log'
            self.algostat['method'] = method
            self.algostat['weight_type'] = weight_type
            num_tri = 0
            maxheap = heapdict()
            if method == 'apprx':
                nbrs = self.G.nbrs 
                num_nodes = len(nbrs)
                nodeset = [v for v in nbrs if len(nbrs[v])>=2 ]
                nu = 100 # 1000 # prob of having good estimate is at least 99%
                eps = 1/math.sqrt(num_nodes) # +-sqrt(n) error will be incurred during tri counting , but with prob at most 1 - ((nu -1)/nu)
                kappa = math.ceil(math.log(2*nu)/(2*eps**2))
                # print('approximate triangle counting: nu = ',nu,' eps = ',eps,' n = ',num_nodes, ' k = ',k)
                V = list(nodeset) # set of nodes whose deg >=2
                absV = len(V)
                # approximate triangle enumeration
                num_tri = 0
                maxheap = heapdict()
                while num_tri < r:
                    for i in range(kappa):
                        j = random.randint(0,absV-1)
                        nbrs_u = nbrs[V[j]]
                        v,w = random.sample(nbrs_u,k=2)
                        if w in nbrs[v]:
                            u = V[j]
                            h_uvw = weightFn((u,v),weight_type) + weightFn((v,w),weight_type) + weightFn((w,u),weight_type)
                            if (u,v,w,u) not in maxheap:
                                maxheap[(u,v,w,u)] = (-h_uvw,num_tri)
                                top_rpaths.append((u,v,w,u))
                                edge_path_index[(u,v)] = edge_path_index.get((u,v),deque())
                                edge_path_index[(u,v)].append(num_tri)
                                edge_path_index[(v,w)] = edge_path_index.get((v,w),deque())
                                edge_path_index[(v,w)].append(num_tri)
                                edge_path_index[(w,u)] = edge_path_index.get((w,u),deque())
                                edge_path_index[(w,u)].append(num_tri)
                                num_tri += 1
                        if num_tri == r:
                            break 
            # exact triangle enumeration -- ver1: enumerate all triangles. Compute triangle-entropies and keep them on a max-heap.
            if method=='opt1':
                for uu in self.G.nbrs:
                    nbr_u = set(self.G.nbrs[uu])
                    if len(nbr_u)<2:    continue 
                    for vv in nbr_u:
                        nbr_v = set(self.G.nbrs[vv])
                        if len(nbr_v) < 2:  continue 
                        tris = nbr_u.intersection(nbr_v)
                        for ww in tris:
                            u,v,w = sorted([uu,vv,ww])
                            if (u,v,w,u) not in maxheap:
                                h_uvw = weightFn((u,v),weight_type) + weightFn((v,w),weight_type) + weightFn((w,u),weight_type)
                                maxheap[(u,v,w,u)] = (h_uvw,num_tri)
                                top_rpaths.append((u,v,w,u))
                                edge_path_index[(u,v)] = edge_path_index.get((u,v),deque())
                                edge_path_index[(u,v)].append(num_tri)
                                edge_path_index[(v,w)] = edge_path_index.get((v,w),deque())
                                edge_path_index[(v,w)].append(num_tri)
                                edge_path_index[(w,u)] = edge_path_index.get((w,u),deque())
                                edge_path_index[(w,u)].append(num_tri)
                                num_tri += 1

            # exact triangle enumeration -- ver2: we keep only top-r triangles in heap based on entropy.
            if method == 'opt2':
                minheap = heapdict()
                edges_with_weights = sorted([(e[0],e[1],weightFn(e,type=weight_type)) for e in self.G.Edges],key=lambda x: -x[2])
                # print(edges_with_weights)
                for uu,vv,wt in edges_with_weights:
                    nbr_u = set(self.G.nbrs[uu])
                    nbr_v = set(self.G.nbrs[vv])
                    if len(nbr_u)>=2 and len(nbr_v)>=2:
                        tris = nbr_u.intersection(nbr_v)
                        for ww in tris:
                            u,v,w = sorted([uu,vv,ww])
                            if (u,v,w,u) not in maxheap:
                                h_uvw = weightFn((u,v),weight_type) + weightFn((v,w),weight_type) + weightFn((w,u),weight_type)
                                if len(maxheap) < r:
                                    minheap[(u,v,w,u)] = (h_uvw,0) # dummy 0
                                    # num_tri += 1
                                else: # kick-out 
                                    minheap.popitem()
                                    minheap[(u,v,w,u)] = (h_uvw, 0) #dummy 0
                                    # num_tri+=1
                for key in minheap:
                    u,v,w,u = key 
                    value = minheap[key]
                    maxheap[key] = (-value[0],num_tri)
                    top_rpaths.append((u,v,w,u))
                    edge_path_index[(u,v)] = edge_path_index.get((u,v),deque())
                    edge_path_index[(u,v)].append(num_tri)
                    edge_path_index[(v,w)] = edge_path_index.get((v,w),deque())
                    edge_path_index[(v,w)].append(num_tri)
                    edge_path_index[(w,u)] = edge_path_index.get((w,u),deque())
                    edge_path_index[(w,u)].append(num_tri)
                    num_tri += 1

            # if verbose: print('|T| = ',num_tri, ' |S|= ',kappa,' |V| = ',absV)
            # print('total #tri: ',num_tri)

            if verbose:
                for key in maxheap:
                    print(key,' => ',maxheap[key])
            if verbose: print('top-r paths: ',top_rpaths)
            # print('top-r path entropies: ',entropy_paths)
                
            count = 0
            round = 0
            structure_len = [0]
            while count<k and len(maxheap):
                if verbose: print('count = ',count, ' len(heap) = ',len(maxheap))
                toppath,(H_p,_index_toppath) = maxheap.popitem()
                if verbose: print(toppath,' => ',H_p)
                indices_of_otherpaths = set() # Because say an alternative path exist containing (u,v) and (v,w) both when top-path = u->v-w. We
                                            # want to avoid duplicates in such cases. 
                toppath_len = len(toppath)
                print('-- ',toppath_len -1 + structure_len[-1], ' ',k)
                if toppath_len -1 + structure_len[-1] >k: # can not add all edges from top-path due to budget exhausted
                    #   Otherwise if the path has more edges than the budget constraint $k$, 
                    #  we select $k$ edges on this path with the highest individual entropy and 
                    # update their probabilities to 1 in order.
                    t = 'hx'
                    edges_in_top_path = [(toppath[j],toppath[j+1],weightFn((toppath[j],toppath[j+1]),t)) \
                                        for j in range(toppath_len-1)]
                    edges_in_top_path.sort(key = lambda x: -x[-1])
                    print('selecting ',k,' edges')
                    while count < k:
                        u,v,edge_entropy = edges_in_top_path[count]
                        print(u,v,' => ',edge_entropy)
                        count+=1
                        E.append((u,v))
                    structure_len.append(count + structure_len[-1])
                else:
                    for j in range(toppath_len-1):
                        u,v = toppath[j],toppath[j+1]
                        E.append((u,v))
                        self.G.edge_update(u,v, type= update_type)
                        # self.G.update_edge_prob(u,v,0.0)
                        self.Query.reset(self.G)
                        count+=1
                        shared_paths = edge_path_index[(u,v)]
                        if len(shared_paths)>1:
                            for others in shared_paths:
                                indices_of_otherpaths.add(others)
                        if property!='tri' and count >= k:   
                            break 
                    structure_len.append(toppath_len -1 +structure_len[-1])
                H_up = self.measure_H0(property, algorithm, T, N, seed = init_seed+round)
                # print('after cleaning ',E[-3:],' H_up = ',H_up)
                history_Hk.append(H_up)
                round += 1
                # Update entropy of any other path that shares an edge with the toppath at current round
                for _index_another_path in indices_of_otherpaths:
                    if _index_another_path != _index_toppath:
                        if property!='tri':
                            another_path = top_rpaths[_index_another_path]
                            h_path = 0
                            for j in range(len(another_path)-1):
                                u,v = another_path[j],another_path[j+1]
                                if (u,v) in self.G.edict:
                                    h_path += h(self.G.get_prob((u,v)))
                                else:
                                    h_path += h(self.G.get_prob((v,u)))
                        else:
                            another_path = top_rpaths[_index_another_path]
                            u,v,w,_ = another_path
                            h_path = weightFn((u,v),weight_type) + weightFn((v,w),weight_type) + weightFn((w,u),weight_type)
                        # if verbose:
                        #     print('update heap: ',another_path, '(before) : ',maxheap[another_path])
                        # maxheap[another_path] = h_path 
                        # if verbose:
                        #     print('update heap: ',another_path, '(after) : ',h_path)
                        if another_path in maxheap:
                            if verbose:
                                print('update heap: ',another_path, '(before) : ',maxheap[another_path])
                            # update priority
                            if property!='tri':
                                if update_type == 'o1': # U1
                                    _count,_ = maxheap[another_path]  # heap priority = ( ordering of the shortest paths generated, -entropy)
                                    maxheap[another_path] = (_count,-h_path) 
                                else: # U2(adaptive/non-adaptive)
                                    _,_count = maxheap[another_path] 
                                    maxheap[another_path] = (-h_path,_count) 
                            else:
                                _,_count = maxheap[another_path] 
                                maxheap[another_path] = (-h_path,_count)
        else:
            raise Exception("invalid query type")
            sys.exit(1)
         
        # H_up = self.measure_H0(property, algorithm, T, N)
        # print('H0  = ',H0)
        # print('H_up = ',H_up)
        # print('reduction: ',H0 - H_up)
        history_Hk = np.array(history_Hk)
        min_hkhat = np.argmin(history_Hk)
        print('E = ',E)
        print('history_Hk: ',history_Hk)
        print('structure_len = ',structure_len)
        print('structure_len[min_hkhat] = ',structure_len[min_hkhat])
        print('min_hkhat = ',min_hkhat)
        # largest_minima_index = np.where(history_Hk[min_hkhat] == history_Hk)[0]
        e_tm = time() - start_execution_time
        self.algostat['MCalgo'] = algorithm
        self.algostat['result']['edges'] = [E[:structure_len[min_hkhat]],[]][min_hkhat==0]
        self.algostat['result']['H*'] = history_Hk[min_hkhat] # self.measure_H0(property,algorithm,T,N, seed = r)
        self.algostat['execution_time']= e_tm
        self.algostat['support'] = ''
        self.algostat['DeltaH'] = self.algostat['result']['H0'] - self.algostat['result']['H*']
        self.algostat['|DeltaH|'] = abs(self.algostat['result']['H0'] - self.algostat['result']['H*'])
        self.algostat['history_Hk'] = ",".join(["{:.4f}".format(i) for i in history_Hk])
        return E,self.algostat['result']['H*'],self.algostat['result']['H0']- self.algostat['result']['H*']

    def compute_Pr_up_zero(self):
        self.Pr_up_0 = {}
        for omega in self.Query.phiInv:
            self.Pr_up_0[omega] = {}
            for e in self.Query.p_graph.edict:
                # p_e = self.G.edict[e] # p(e)
                _sumPrG = 0
                for i in self.Query.phiInv[omega]:
                    if e in self.Query.index[i]: # if edge e is in G_i where omega is observed
                        pass
                        # p_e = self.Query.hatp[e] # using \hat{p}(e) instead of p(e) before update
                        # p_up_e = 1 # p_up(e)
                        # G_up = self.Query.PrG[i] * p_up_e / p_e  # Pr_up(G_i) = Pr(G) * p_up(e) / p(e)
                        # _sumPrG += G_up 
                    else: #  edge e not in G_i where omega is observed
                        # if p_e != p_up_e:
                        try:
                            p_e = self.Query.hatp[e]
                            G_up = self.Query.PrG[i] / (1-p_e)  # Pr_up(G_i) = Pr(G_i) * (1- p_up(e)) / (1- p(e))
                            _sumPrG += G_up 
                        except ZeroDivisionError:
                            print(p_e,self.Query.PrG[i])
                            raise ZeroDivisionError
                                
                self.Pr_up_0[omega][e] = _sumPrG

    def compute_approx_dhreach(self, s,t,d,probGraph, numsamples):
        random.seed()
        sid = random.randint(0,1000000)
        func_obj = probGraph.get_Ksample_dhopbfs(K = numsamples,seed=sid,\
                                    source=s,target = t, dhop = d, optimiseargs = None)
        hat_Pr = {}
        for _,_, _,omega in func_obj:
            hat_Pr[omega] = hat_Pr.get(omega,0) + (1.0/numsamples)
        # print(hat_Pr)
        reachability = hat_Pr.get(1,0)
        return reachability
    
    def crowd_kbest(self, property, algorithm, k, K, update_dict, N=1,T=1, update_type = 'c1', verbose = False, track_H = False):
        """ Greedy with mem. in crowdsourced setting """
        print(['adaptive','non-adaptive'][update_type == 'c1'], ' setting')
        assert k>=1
        history = []
        start_execution_time = time()
        if verbose: print('Query eval.')
        self.algostat['result']['H0'] = self.measure_H0(property,algorithm,T,N)
        # Start of Algorithm 5
        print(self.Query.qtype)
        self.Query.constructTables_S(K = K, verbose = verbose)
        Estar = copy(self.G.edict)
        assert k<=len(Estar)
        E = []
        if (verbose):
            print('H0: ',self.algostat['result']['H0'])
            print('p(e): ',self.G.edict)
            print('\hatp(e): ',self.Query.hatp)
            print('Pr[G]: ',self.Query.PrG)
            print('G: ',self.Query.G)
            # print('Pr[Omega]: ', self.Query.freq_distr)
            # print('results: ',self.Query.results)
        # Calculate Pr_up^{e_j}(G_i) for all i,j
        self.compute_Pr_up(verbose = verbose)
        self.compute_Pr_up_zero()
        can_reduce_further = True 
        history_Hk = []
        previous_pe = []
        for iter in range(k):
            if not can_reduce_further:
                break 
            if (verbose):
                print("Iteration: ",iter)
            # Start of Algorithm 4
            # print('Pr(G_i): ', self.Query.PrG)
            # Construct Pr[Omega] 
            Pr_Omega = {}
            for omega in self.Query.phiInv:
                Pr_Omega[omega] = sum([self.Query.PrG[i] for i in self.Query.phiInv[omega]])
            H0_i = entropy([j for i,j in Pr_Omega.items()], base = 2)
            if self.debug: print('Pr[omega]: ',Pr_Omega)
            if self.debug:  print("M before update: \n\r",self.Pr_up)
            if self.debug:  M = deepcopy(self.Pr_up)#; print(M)
            if (verbose):
                # self.Query.eval()
                H0 = self.measure_H0(property,algorithm,T,N)
                # print('H0: ',H0)

            if (verbose):
                print('Selecting edge#: ',iter+1)
                # print('Jacobian: ', DelPr_Omega_pe)
                print('Pr[Omega]: ',Pr_Omega)
                print("Pr_up[Omega]: ",self.Pr_up)
                print('Gsums: ',self.Gsums)
            # local_maxima = None 
            minima = []
            candidates = []
            for e in Estar: # Among remaining edges (Estar), find the one (e*) with largest  H - H_{up}
                DeltaHe2 = 0 # H - H^{e}_{up}
                H_up = 0
                p_e = self.G.get_prob(e)
                for omega in self.Query.phiInv:
                    # if Pr_Omega[omega] == 0:
                    #     h0 = 0
                    # else:
                    #     h0 = log2(Pr_Omega[omega])*Pr_Omega[omega]

                    if self.Pr_up[omega].get(e,0) == 0:
                        h1 = 0
                    else:
                        h1 = p_e*log2(self.Pr_up[omega][e])*self.Pr_up[omega][e]
                    if self.Pr_up_0[omega].get(e,0) == 0:
                        h2 = 0
                    else:
                        h2 = (1-p_e)*log2(self.Pr_up_0[omega][e])*self.Pr_up_0[omega][e]
                    expected_entropy = h1 + h2
                    # DeltaHe2 += (expected_entropy - h0)
                    H_up += -expected_entropy
                minima.append(H_up)
                candidates.append(e)
                # if verbose:
                if self.debug:
                    print('H_up^e', e,' -> ',H_up)
                    # print('e: ',e,' DeltaH2[e]: ',DeltaHe2)
                    # print('H_next [',e,']: ', self.algostat['result']['H0'] + DeltaHe2)
                
                # if local_maxima is not None:
                #     if DeltaHe2 > local_maxima[1]:
                #         local_maxima = (e,DeltaHe2)
                # else:
                #    local_maxima = (e,DeltaHe2)
            minima_index = np.argmin(minima)
            local_maxima = (candidates[minima_index],H0_i - minima[minima_index])
            if verbose: print('e*: ',local_maxima[0],' DeltaH2[e]: ',local_maxima[1])
            estar = local_maxima[0] # assign e*
            # End of Algorithm 4
            E.append(estar)
            del Estar[estar] # Delete e* from dictionary s.t. next iteration is 1 less than the current.
            history_Hk.append(minima[minima_index])
            previous_pe.append(self.G.get_prob(estar))
            if update_type == 'c2':
                if iter < k-1:
                    can_reduce_further = self.Query.adaptiveUpdateTables(estar,update_dict[estar], self.Pr_up,self.Pr_up_0)
                self.G.update_edge_prob(estar[0],estar[1],update_dict[estar])
                # print('-',self.Query.PrG)
                self.Query.reset(self.G) # Re-initialise Query() with updated UGraph()  
                # history.append(self.algostat['result']['H0']-self.measure_H0(property,algorithm,T,N)) 
            
            if (verbose):
                print('----------')
        if update_type == 'c1':
            for e in E:
                self.G.update_edge_prob(e[0],e[1],update_dict[e])
            self.Query.reset(self.G) # Re-initialise Query() with updated UGraph()  
            if track_H:
                history.append(self.algostat['result']['H0']-self.measure_H0(property,algorithm,T,N))
        history_Hk = np.array(history_Hk)
        min_hkhat = np.argmin(history_Hk)
        minima_index_list = np.where(history_Hk[min_hkhat] == history_Hk)[0]
        largest_minima_index = minima_index_list[-1]
        for i in range(largest_minima_index+1,len(E)):
            self.Query.p_graph.update_edge_prob(E[i][0],E[i][1],previous_pe[i])
        E = E[:largest_minima_index+1]
        e_tm = time() - start_execution_time
        self.algostat['algorithm'] = 'greedy+mem'
        self.algostat['MCalgo'] = algorithm
        self.algostat['k'] = k
        self.algostat['result']['edges'] = E
        # self.Query.eval()
        # self.algostat['result']['H*'] = self.algostat['result']['H0']-history[-1]
        self.algostat['result']['H*'] = self.measure_H0(property,algorithm,T,N)
        self.algostat['execution_time'] = e_tm
        # self.algostat['support'] = str(list(self.Query.phiInv.keys()))
        self.algostat['DeltaH'] = self.algostat['result']['H0'] - self.algostat['result']['H*']
        self.algostat['|DeltaH|'] = abs(self.algostat['result']['H0'] - self.algostat['result']['H*'])
        self.algostat['history_deltaH'] = history
        if self.debug: self.algostat['M'] = M
        return E,self.algostat['result']['H*'], self.algostat['result']['H0']- self.algostat['result']['H*']

    def crowd_greedyp(self, property, algorithm, k, K, r, update_dict, N=1,T=1, update_type = 'c1', verbose = False, track_H = False):
        """ Greedy+p in crowdsourced setting """
        # N = 2*N
        # T = 2*T 
        ZERO = 10**(-13)
        def h(x):
            absx = abs(x)
            if absx <= ZERO:
                return 0
            elif (1-absx) <= ZERO:
                return 0
            else:
                try:
                    p = log2(x)
                except:
                    print(x)
            return -x*log2(x)
        
        def weightFn(e, type = 'log'): # type = `log` (-p(e)logp(e)), `hx` (H(e)), None (1), orig (w(e))
            if type is None:
                return 1
            else:
                x = self.G.get_prob(e)
                if type == 'log':
                    return h(x) # -p(e)log p(e)
                elif type == 'hx':
                    return h(x)+h(1-x) # Entropy(e)
                elif type == 'hx_cr':
                    h_x = h(x)
                    h_1_x = (1-h_x)
                    return x*h_x + (1-x)*h_x+x*h_1_x + (1-x)*h_1_x # p(e) Entropy(e) + (1-p(e))*(1-ENtropy(e))
                elif type == 'orig':
                    return self.G.weights[e]
        def path_entropy(path):
            path_prob = 1
            for j in range(len(path)-1):
                u,v = path[j],path[j+1]
                if (u,v) in self.G.edict:
                    path_prob *= self.G.get_prob((u,v))
                else:
                    path_prob *= self.G.get_prob((v,u))
            return h(path_prob)+h(1-path_prob)
        
        def compute_approx_reach(s,t,d,probGraph, N, T, seed):
            numsamples = N*T
            # numsamples = T
            # print('num of samples: ',numsamples)
            func_obj = probGraph.get_Ksample_dhopbfs(K = numsamples,seed=seed,\
                                        source=s,target = t, dhop = d, optimiseargs = None)
            hat_Pr = {}
            for _,_, _,omega in func_obj:
                hat_Pr[omega] = hat_Pr.get(omega,0) + (1.0/numsamples)
            return hat_Pr.get(1,0)

        def d_hop_entropy(seed):
            r = compute_approx_reach(self.Query.u,self.Query.v,self.Query.d,self.G, N,T,seed)
            return h(r) + h(1-r)
        
        print(['adaptive','non-adaptive'][update_type == 'c1'], ' setting')
        assert k>=1 and update_type!='o1'
        if property!='tri':
            init_seed = int(str(self.Query.u)+str(self.Query.v))
        if property=='reach_d':
            H0 = d_hop_entropy(1)
            # H0 = self.measure_H0(property,algorithm,T,N, seed = 1)
        else:
            H0 = self.measure_H0(property,algorithm,T,N, seed = 1)

        self.algostat['result']['H0'] = H0

        start_execution_time = time()
        E = []
        history_Hk = [H0]
        top_rpaths = []
        edge_path_index = {}
        # Step 1: Top-r Structure selection
        if property!='tri':
            nx_G = self.G.nx_format
            # edges_with_weights = [(e[0],e[1],weightFn(e,type=None)) for e in self.G.Edges]
            # nx_G.add_weighted_edges_from(edges_with_weights)
            nx_G.add_edges_from(self.G.Edges)
            if verbose: print(nx_G.edges(data=True))
            # entropy_paths = []
            count = 0
            # for path in nx.all_simple_paths(nx_G,self.Query.u,self.Query.v):
            #     print(path)
            # path_gen = nx.all_simple_paths(nx_G,self.Query.u,self.Query.v)
            path_gen = nx.all_shortest_paths(nx_G,self.Query.u,self.Query.v,weight='weight')
            maxheap = heapdict()
            for path in path_gen:
                if verbose: print(path)
                if count< r:
                    for j in range(len(path)-1):
                        u,v = path[j],path[j+1]
                        edge_path_index[(u,v)] = edge_path_index.get((u,v),deque())
                        edge_path_index[(u,v)].append(count)
                    h_path = path_entropy(path)
                    maxheap[tuple(path)] = (-h_path,count) # heap priority = (-entropy, ordering of the shortest paths generated)
                    top_rpaths.append(tuple(path))
                    count+=1
                else:
                    break
        elif property == 'tri':
            weight_type = 'log'
            minheap = heapdict()
            edges_with_weights = sorted([(e[0],e[1],weightFn(e,type=weight_type)) for e in self.G.Edges],key=lambda x: -x[2])
            # print(edges_with_weights)
            for uu,vv,wt in edges_with_weights:
                nbr_u = set(self.G.nbrs[uu])
                nbr_v = set(self.G.nbrs[vv])
                if len(nbr_u)>=2 and len(nbr_v)>=2:
                    tris = nbr_u.intersection(nbr_v)
                    for ww in tris:
                        u,v,w = sorted([uu,vv,ww])
                        if (u,v,w,u) not in maxheap:
                            h_uvw = weightFn((u,v),weight_type) + weightFn((v,w),weight_type) + weightFn((w,u),weight_type)
                            if len(maxheap) < r:
                                minheap[(u,v,w,u)] = (h_uvw,0) # dummy 0
                                # num_tri += 1
                            else: # kick-out 
                                minheap.popitem()
                                minheap[(u,v,w,u)] = (h_uvw, 0) #dummy 0
            for key in minheap:
                u,v,w,u = key 
                value = minheap[key]
                maxheap[key] = (-value[0],num_tri)
                top_rpaths.append((u,v,w,u))
                edge_path_index[(u,v)] = edge_path_index.get((u,v),deque())
                edge_path_index[(u,v)].append(num_tri)
                edge_path_index[(v,w)] = edge_path_index.get((v,w),deque())
                edge_path_index[(v,w)].append(num_tri)
                edge_path_index[(w,u)] = edge_path_index.get((w,u),deque())
                edge_path_index[(w,u)].append(num_tri)
                num_tri += 1
            # # print(self.G.nbrs)
            # nbrs = self.G.nbrs 
            # num_nodes = len(nbrs)
            # nodeset = [v for v in nbrs if len(nbrs[v])>=2 ]
            # nu = 100 # 1000 # prob of having good estimate is at least 99%
            # eps = 1/math.sqrt(num_nodes) # +-sqrt(n) error will be incurred during tri counting , but with prob at most 1 - ((nu -1)/nu)
            # kappa = math.ceil(math.log(2*nu)/(2*eps**2))
            # # print('approximate triangle counting: nu = ',nu,' eps = ',eps,' n = ',num_nodes, ' k = ',k)
            # V = list(nodeset) # set of nodes whose deg >=2
            # absV = len(V)
            # # approximate counting
            # num_tri = 0
            # maxheap = heapdict()
            # while num_tri < r:
            #     for i in range(kappa):
            #         j = random.randint(0,absV-1)
            #         nbrs_u = nbrs[V[j]]
            #         v,w = random.sample(nbrs_u,k=2)
            #         if w in nbrs[v]:
            #             u = V[j]
            #             h_uvw = weightFn((u,v),'log') + weightFn((v,w),'log') + weightFn((w,u),'log')
            #             maxheap[(u,v,w,u)] = (-h_uvw,num_tri)
            #             top_rpaths.append((u,v,w,u))
            #             edge_path_index[(u,v)] = edge_path_index.get((u,v),deque())
            #             edge_path_index[(u,v)].append(num_tri)
            #             edge_path_index[(v,w)] = edge_path_index.get((v,w),deque())
            #             edge_path_index[(v,w)].append(num_tri)
            #             edge_path_index[(w,u)] = edge_path_index.get((w,u),deque())
            #             edge_path_index[(w,u)].append(num_tri)
            #             num_tri += 1
            #         if num_tri == r:
            #             break 
            # if verbose: print('|T| = ',num_tri, ' |S|= ',kappa,' |V| = ',absV)
        else:
            raise Exception("invalid query type")
            sys.exit(1)

        if verbose: print('top-r paths: ',top_rpaths)
        # print('top-r path entropies: ',entropy_paths)
            
        count = 0
        round = 0
        structure_len = [0]
        while count < k and len(maxheap):
            if verbose: print('count = ',count, ' len(heap) = ',len(maxheap))
            toppath,(H_p,_index_toppath) = maxheap.popitem()
            indices_of_otherpaths = set() # Because say an alternative path exist containing (u,v) and (v,w) both when top-path = u->v-w. We
                                          # want to avoid duplicates in such cases. 
            toppath_len = len(toppath)
            # print('-- ',toppath_len -1 + structure_len[-1], ' ',k)
            if toppath_len -1 + structure_len[-1] >k: # can not add all edges from top-path due to budget exhausted
                #   Otherwise if the path has more edges than the budget constraint $k$, 
                #  we select $k$ edges on this path with the highest individual entropy and 
                # update their probabilities to 1 in order.
                t = 'hx'
                edges_in_top_path = [(toppath[j],toppath[j+1],weightFn((toppath[j],toppath[j+1]),t)) \
                                    for j in range(toppath_len-1)]
                edges_in_top_path.sort(key = lambda x: -x[-1])
            
                # print('selecting ',k,'-best edges')
                for u,v,edge_entropy in edges_in_top_path:
                    if count>=k:
                        break
                    # u,v,edge_entropy = edges_in_top_path[count]
                    # print(u,v,' => ',edge_entropy)
                    count+=1
                    E.append((u,v))
                    if update_type == 'c2':
                        # self.G.edge_update(u,v, type= update_type)
                        if (u,v) in update_dict:
                            cr_pe = update_dict[(u,v)]
                        else:
                            cr_pe = update_dict[(v,u)]
                        # print((u,v),' => ',cr_pe,' /',self.G.get_prob((u,v)))
                        self.G.update_edge_prob(u,v,cr_pe)
                        self.Query.reset(self.G)
                structure_len.append(count + structure_len[-1])
            else:
                for j in range(toppath_len-1):
                    u,v = toppath[j],toppath[j+1]
                    E.append((u,v))
                    if update_type == 'c2':
                        # self.G.edge_update(u,v, type= update_type)
                        if (u,v) in update_dict:
                            cr_pe = update_dict[(u,v)]
                        else:
                            cr_pe = update_dict[(v,u)]
                        # print((u,v),' => ',cr_pe,' /',self.G.get_prob((u,v)))
                        self.G.update_edge_prob(u,v,cr_pe)
                        self.Query.reset(self.G)
                        # print('--- ',self.G.get_prob((u,v)))
                        shared_paths = edge_path_index[(u,v)]
                        if len(shared_paths)>1:
                            for others in shared_paths:
                                indices_of_otherpaths.add(others)
                    count+=1
                    if count >= k:   break 
                structure_len.append(toppath_len -1 +structure_len[-1])
            round += 1
            if update_type == 'c2':
                if len(E) == k:
                    sid = 2*int(str(self.Query.u)+str(self.Query.v))
                else:
                    sid = init_seed+round
                if property=='reach_d':
                    H_up = d_hop_entropy(sid)
                    # H_up = self.measure_H0(property,algorithm,T,N, seed = sid)
                else:
                    H_up = self.measure_H0(property,algorithm,T,N, seed = sid)
                # H_up = self.measure_H0(property, algorithm, T, N, seed = sid)
                history_Hk.append(H_up)
                print('round ',round,' => H_up = ',H_up)
            
            if update_type == 'c1':
                assert len(indices_of_otherpaths) == 0
            # Update entropy of any other path that shares an edge with the toppath at current round
            for _index_another_path in indices_of_otherpaths:
                if _index_another_path != _index_toppath:
                    if property!='tri':
                        another_path = top_rpaths[_index_another_path]
                        h_path = path_entropy(another_path)
                    else:
                        another_path = top_rpaths[_index_another_path]
                        u,v,w,_ = another_path
                        h_path = weightFn((u,v),weight_type) + weightFn((v,w),weight_type) + weightFn((w,u),weight_type)

                    if another_path in maxheap:
                        if verbose:
                            print('update heap: ',another_path, '(before) : ',maxheap[another_path])
                        # update priority
                        # if property!='tri':
                             # U2(adaptive/non-adaptive)
                        _,_count = maxheap[another_path] 
                        maxheap[another_path] = (-h_path,_count) 
                        # else:
                        #     _,_count = maxheap[another_path] 
                        #     maxheap[another_path] = (-h_path,_count) 
                        if verbose:
                            print('update heap: ',another_path, '(after) : ',maxheap[another_path])
        print('#rounds = ',round)
        if update_type == 'c1':
            for e in E:
                if (u,v) in update_dict:
                    cr_pe = update_dict[(u,v)]
                else:
                    cr_pe = update_dict[(v,u)]
                self.G.update_edge_prob(e[0],e[1],cr_pe)
            self.Query.reset(self.G) # Re-initialise Query() with updated UGraph()  
            H_up = self.measure_H0(property, algorithm, T, N, seed = 2*int(str(self.Query.u)+str(self.Query.v)))
            history_Hk.append(H_up)
        history_Hk = np.array(history_Hk)
        min_hkhat = np.argmin(history_Hk)
        print('E = ',E)
        print('history_Hk: ',history_Hk)
        print('structure_len = ',structure_len)
        print('structure_len[min_hkhat] = ',structure_len[min_hkhat])
        print('min_hkhat = ',min_hkhat)

        e_tm = time() - start_execution_time
        self.algostat['algorithm'] = 'greedy+struct'
        self.algostat['MCalgo'] = algorithm
        self.algostat['k'] = k
        self.algostat['result']['edges'] = [E[:structure_len[min_hkhat]],[]][min_hkhat==0]# E
        self.algostat['result']['H*'] =  history_Hk[min_hkhat]
        self.algostat['execution_time'] = e_tm
        # self.algostat['support'] = str(list(self.Query.phiInv.keys()))
        self.algostat['DeltaH'] = self.algostat['result']['H0'] - self.algostat['result']['H*']
        self.algostat['|DeltaH|'] = abs(self.algostat['result']['H0'] - self.algostat['result']['H*'])
        self.algostat['history_Hk'] = ",".join(["{:.4f}".format(i) for i in history_Hk])
        return E,self.algostat['result']['H*'], self.algostat['result']['H0']- self.algostat['result']['H*']


    def compute_approx_reach(self, s,t,probGraph, numsamples):
        random.seed()
        sid = random.randint(0,1000000)
        func_obj = probGraph.get_Ksample_bfs(K = numsamples,seed=sid,\
                                    source=s,target = t, optimiseargs = None)
        hat_Pr = {}
        for _,_, _,omega in func_obj:
            hat_Pr[omega] = hat_Pr.get(omega,0) + (1.0/numsamples)
        # print(hat_Pr)
        reachability = hat_Pr.get(1,0)
        return reachability
    
    def crowd_kbest_greedy(self, property, algorithm, k, update_dict, N=1,T=1, update_type = 'c1', verbose = False):
        """ Greedy without mem. in crowdsourced setting """
        ZERO = 10**(-13)
        def h(x):
            absx = abs(x)
            if absx <= ZERO:
                return 0
            elif (1-absx) <= ZERO:
                return 0
            else:
                try:
                    p = log2(x)
                except:
                    print(x)
            return -x*log2(x)
        print(['adaptive','non-adaptive'][update_type == 'c1'], ' setting')
        assert k>=1
        history = []
        self.algostat['algorithm'] = 'greedy'
        self.algostat['k'] = k
        start_execution_time = time()
        Estar = copy(self.G.edict)
        self.algostat['result']['H0'] = self.measure_H0(property,algorithm,T,N)
        assert k<=len(Estar)
        E = []
        H0 = self.algostat['result']['H0']
        # Calculate Pr_up^{e_j}(G_i) for all i,j
        # self.compute_Pr_up(verbose = verbose)
        # self.compute_Pr_up_zero()
        for iter in range(k):
            if (verbose):
                print("Iteration: ",iter)

            local_maxima = None 
            for e in Estar: # Among remaining edges (Estar), find the one (e*) with largest  H - H_{up}
                g_copy = deepcopy(self.G)
                pe = self.G.get_prob(e)
                if property == 'reach':
                    # Reach = self.compute_approx_reach(self.Query.u,self.Query.v,\
                    #                                    self.Query.p_graph, N*T)
                    g_copy.update_edge_prob(u = e[0],v = e[1], prob = 0)
                    self.Query.reset(g_copy)
                    PrOmega_1 = self.compute_approx_reach(self.Query.u,self.Query.v,\
                                                       self.Query.p_graph, N*T)
                    g_copy.update_edge_prob(u = e[0],v = e[1], prob = 1)
                    self.Query.reset(g_copy)
                    PrOmega_0 = self.compute_approx_reach(self.Query.u,self.Query.v,\
                                                       self.Query.p_graph, N*T)
                    h1 = h(PrOmega_1) + h(1-PrOmega_1)
                    h0 = h(PrOmega_0) + h(1-PrOmega_0)
                    expected_entropy = pe*h0 + (1-pe)*h1

                elif property == 'reach_d':
                    # Reach = self.compute_approx_reach(self.Query.u,self.Query.v,\
                    #                                    self.Query.p_graph, N*T)
                    g_copy.update_edge_prob(u = e[0],v = e[1], prob = 0)
                    self.Query.reset(g_copy)
                    PrOmega_1 = self.compute_approx_dhreach(self.Query.u,self.Query.v,self.Query.d,\
                                                       self.Query.p_graph, N*T)
                    g_copy.update_edge_prob(u = e[0],v = e[1], prob = 1)
                    self.Query.reset(g_copy)
                    PrOmega_0 = self.compute_approx_dhreach(self.Query.u,self.Query.v,self.Query.d,\
                                                       self.Query.p_graph, N*T)
                    h1 = h(PrOmega_1) + h(1-PrOmega_1)
                    h0 = h(PrOmega_0) + h(1-PrOmega_0)
                    expected_entropy = pe*h0 + (1-pe)*h1 
                else:
                    g_copy.update_edge_prob(u = e[0],v = e[1], prob = 0)
                    self.Query.reset(g_copy)
                    h0 = self.measure_H0(property, algorithm, T, N)
                    g_copy.update_edge_prob(u = e[0],v = e[1], prob = 1)
                    self.Query.reset(g_copy)
                    h1 = self.measure_H0(property, algorithm, T, N)
                    expected_entropy = pe*h0 + (1-pe)*h1
                    
                if local_maxima is not None:
                    if H0 - expected_entropy > H0 - local_maxima[1]:
                        # DeltaHe = H0 - expected_entropy
                        local_maxima = (e, expected_entropy)
                else:
                    local_maxima = (e, expected_entropy)
            estar = local_maxima[0] # assign e*
            # H0 = local_maxima[1] # Assign H0 for the next iteration.
            E.append(estar)
            del Estar[estar] # Delete e* from dictionary s.t. next iteration is 1 less than the current.
            if update_type == 'c2':
                self.G.update_edge_prob(estar[0],estar[1],update_dict[estar])
                self.Query.reset(self.G) # Re-initialise Query() with updated UGraph()  
                # history.append(self.algostat['result']['H0']-self.measure_H0(property,algorithm,T,N)) 
                # if update_dict[estar] == 0:
                #     H0 = h0 # H0 for the edge selection
                # else:
                #     H0 = h1 # H0 for next edge selection

        if update_type == 'c1':
            for e in E:
                self.G.update_edge_prob(e[0],e[1],update_dict[e])
                self.Query.reset(self.G) # Re-initialise Query() with updated UGraph()  
                # history.append(self.algostat['result']['H0']-self.measure_H0(property,algorithm,T,N))

        e_tm = time() - start_execution_time
        self.algostat['algorithm'] = 'greedy'
        self.algostat['MCalgo'] = algorithm
        self.algostat['k'] = k
        self.algostat['result']['edges'] = E
        # self.Query.eval()
        # self.algostat['result']['H*'] = self.algostat['result']['H0']-history[-1]
        self.algostat['result']['H*'] = self.measure_H0(property,algorithm,T,N)
        self.algostat['execution_time'] = e_tm
        # self.algostat['support'] = str(list(self.Query.phiInv.keys()))
        self.algostat['DeltaH'] = self.algostat['result']['H0'] - self.algostat['result']['H*']
        self.algostat['|DeltaH|'] = abs(self.algostat['result']['H0'] - self.algostat['result']['H*'])
        self.algostat['history_deltaH'] = history
        return E,self.algostat['result']['H*'], self.algostat['result']['H0']- self.algostat['result']['H*']
        # Test: python reduce_main_crowd.py -k 1 -pr reach -a greedy -ea mcbfs -d products -q data/queries/products/products_2.queries -cr data/large/crowd/product_pair.true -mq 1 -dh 0 &
