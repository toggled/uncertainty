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

def save_pickle(ob, fname):
    with open (fname, 'wb') as f:
        #Use the dump function to convert Python objects into binary object files
        pickle.dump(ob, f)

def load_pickle(fname):
    with open (fname, 'rb') as f:
        #Convert binary object to Python object
        ob = pickle.load(f)
        return ob

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
        self.G = g 
        assert isinstance(self.G, Graph)
        # assert isinstance(query, Query)
        self.Query = query
        self.algostat['execution_time'] = 0
        self.algostat['result'] = {}
        self.algostat['k'] = 0
        self.algostat['support'] = [] # Since all sup values may not be observed in the sample, it is good to record those observed.
        self.algostat['algorithm'] = ''
        # self.algostat['peak_memB'] = []
    
    def measure_uncertainty(self, N=1, T=10):
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
                    hat_Pr = {}
                    j = 0
                    if 'time_seed' in os.environ:
                        random.seed()
                        s = random.randint(0,1000000)
                    else:
                        s = i
                    # print('seed: ',s)
                    for g in self.G.get_Ksample(T,seed=s):
                        start_tm = time()
                        omega = self.Query.evalG(g[0])
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
                        bucket_distr = {i: 0 for i in range(index_ub)}
                    for s in S:
                        omega_i = (1.0/T)
                        _index = math.floor((s - _min)/delta)
                        bucket_distr[_index] += omega_i
                    hat_Pr = bucket_distr
                    # print([(self.buckets[_index],bucket_distr[_index]) for _index in bucket_distr])
                else: # non-bucketing & no-pre
                    hat_Pr = {}
                    j = 0
                    if 'time_seed' in os.environ:
                        random.seed()
                        s = random.randint(0,1000000)
                    else:
                        s = i
                    for g in self.G.get_Ksample(T,seed=s):
                        start_tm = time()
                        omega = self.Query.evalG(g[0])
                        query_evaluation_times.append(time()-start_tm)
                        hat_Pr[omega] = hat_Pr.get(omega,0) + 1.0/T
                        support_observed.append(omega)
                        if os.environ['precomp']:
                            save_pickle(omega, os.path.join(os.environ['precomp'],str(i)+"_"+str(j)+".pre"))
                            j+=1
                hat_H = -sum([hat_Pr[omega] * log2(hat_Pr[omega]) for omega in hat_Pr])
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
                    print(support_observed)
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
        source,target = self.Query.u, self.Query.v
        start_execution_time = time()
        query_evaluation_times = []
        hat_H_list = []
        support_observed = []
        sum_H = 0
        if optimise:
            precomputed_nbrs_path = os.path.join('_'.join(os.environ['precomp'].split('_')[:-2])+"_nbr.pre")
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
                if not os.path.isfile(precomputed_nbrs_path):
                    # print('saving nbrs file for the 1st time.')
                    save_pickle(nbrs,precomputed_nbrs_path)
            # print(entropy([hat_Pr[1] ,hat_Pr[0]]))
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

    def measure_uncertainty_bfs(self, N=1, T=10, optimise = False):
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
                    s = i
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
    
    def measure_uncertainty_dhopbfs(self, d = 1, N=1, T=10):
        """
        MC + d-hop BFS 
        """
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
                s = i
            hat_Pr = {}
            func_obj = self.G.get_Ksample_dhopbfs(K = T,seed=s,\
                                        source=source,target = target, dhop = d, optimiseargs = None)
            for _,_, _,omega in func_obj:
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

    def measure_uncertainty_dij(self, N=1, T=10, optimise = False):
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
                precomputed_nbrs_path = os.path.join('_'.join(os.environ['precomp'].split('_')[:-2])+"_nbr.pre")
                if optimise and os.path.isfile(precomputed_nbrs_path):
                    # print('loading precomputed nbrs file..')
                    loaded_nbrs = load_pickle(precomputed_nbrs_path)
                    func_obj = self.G.get_Ksample_dij(T,seed=i,source=source,target = target, optimiseargs = \
                                                      {'nbrs':loaded_nbrs,'doopt': True})
                else:
                    func_obj = self.G.get_Ksample_dij(T,seed=i,source=source,target = target, optimiseargs = None)
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
                    hat_Pr = {}
                    j = 0
                    for g in self.G.get_Ksample(T,seed=i):
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
                    hat_Pr = {}
                    j = 0
                    for g in self.G.get_Ksample(T,seed=i):
                        start_tm = time()
                        omega = self.G.count_tri_approx(g[0])
                        query_evaluation_times.append(time()-start_tm)
                        hat_Pr[omega] = hat_Pr.get(omega,0) + 1.0/T
                        support_observed.append(omega)
                        if os.environ['precomp']:
                            save_pickle(omega, os.path.join(os.environ['precomp'],str(i)+"_"+str(j)+".pre"))
                            j+=1
                hat_H = -sum([hat_Pr[omega] * log2(hat_Pr[omega]) for omega in hat_Pr])
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
                        
    def measure_H0(self, property, algorithm, K, N = 1):
        if algorithm == 'exact':
            self.Query.eval()
            H = self.Query.compute_entropy() # Initial entropy. Kept outside of time() because H0 is not needed in contr table and computed only for logging.
        elif algorithm == 'appr':
            H = self.measure_uncertainty(N=N, T = K)
        elif (algorithm == 'mcbfs' or algorithm == 'pTmcbfs'):
            if property == 'reach':
                assert (property == 'reach')
                H = self.measure_uncertainty_bfs(N=N, T = K)
            else:
                assert (property == 'reach_d')
                H = self.measure_uncertainty_dhopbfs(d = self.Query.d, N = N, T = K)
        elif algorithm == 'mcdij':
            assert (property == 'sp')
            H =  self.measure_uncertainty_dij(N=N, T = K)
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
    
    def algorithm5(self, property, algorithm, k, K, N=1,T=1, update_type = 'o1', verbose = False):
        """ Variant of Algorithm 5 where CT is approximated via sampling"""
        assert k>=1
        history = []
        start_execution_time = time()
        if verbose: print('Query eval.')
        self.algostat['result']['H0'] = self.measure_H0(property,algorithm,T,N)
        # Start of Algorithm 5
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
        for iter in range(k):
            if (verbose):
                print("Iteration: ",iter)
            # Start of Algorithm 4
            # print('Pr(G_i): ', self.Query.PrG)
            # Construct Pr[Omega] 
            Pr_Omega = {}
            for omega in self.Query.phiInv:
                Pr_Omega[omega] = sum([self.Query.PrG[i] for i in self.Query.phiInv[omega]])
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
            local_maxima = None 
            for e in Estar: # Among remaining edges (Estar), find the one (e*) with largest  H - H_{up}
                DeltaHe2 = 0 # H - H^{e}_{up}
                H_up = 0
                for omega in self.Query.phiInv:
                    if (Pr_Omega[omega] != 0):
                        if (self.Pr_up[omega].get(e,0) != 0):
                            # print(e,omega,' =>',self.Pr_up[omega][e])
                            # print(Pr_Omega[omega])
                            DeltaHe2 += (log2(self.Pr_up[omega][e])*self.Pr_up[omega][e] - log2(Pr_Omega[omega])*Pr_Omega[omega])
                            H_up += (-log2(self.Pr_up[omega][e])*self.Pr_up[omega][e])
                        else:
                            DeltaHe2 += (0 - log2(Pr_Omega[omega])*Pr_Omega[omega])
                    else:
                        if (self.Pr_up[omega].get(e,0) != 0):
                            DeltaHe2 += (log2(self.Pr_up[omega][e])*self.Pr_up[omega][e] - 0)
                        else:
                            DeltaHe2 += 0
                # if verbose:
                if self.debug:
                    print('H_up^e', e,' -> ',H_up)
                    print('e: ',e,' DeltaH2[e]: ',DeltaHe2)
                    # print('H_next [',e,']: ', self.algostat['result']['H0'] + DeltaHe2)
                
                if local_maxima is not None:
                    if DeltaHe2 > local_maxima[1]:
                        local_maxima = (e,DeltaHe2)
                else:
                   local_maxima = (e,DeltaHe2)
            if verbose: print('e*: ',local_maxima[0],' DeltaH2[e]: ',local_maxima[1])
            estar = local_maxima[0] # assign e*
            # End of Algorithm 4
            E.append(estar)
            del Estar[estar] # Delete e* from dictionary s.t. next iteration is 1 less than the current.
            if iter < k-1:
                self.Query.updateTables(estar, self.Pr_up)
            else:
                if self.debug:
                    self.Query.updateTables(estar, self.Pr_up)
                    # self.G.edge_update(estar[0], estar[1], type = update_type)
                    # self.Query.reset(self.G)
                    # self.compute_Pr_up(update_type, verbose = verbose)
                    print('-----')
                    print('M after update: ',self.Pr_up)

            self.G.edge_update(estar[0],estar[1],type = update_type) # Update UGraph()
            # print('-',self.Query.PrG)
            self.Query.reset(self.G) # Re-initialise Query() with updated UGraph()  
            history.append(self.algostat['result']['H0']-self.measure_H0(property,algorithm,T,N)) 
            # print('=',self.Query.PrG)         
            if (verbose):
                print('After p(e) update: ')
                print('Pr(G_i): ', self.Query.PrG)
                print('\hatp(e): ',self.Query.hatp)
                # print('C = ',self.Query.C)
            if (verbose):
                print('----------')
        e_tm = time() - start_execution_time
        self.algostat['algorithm'] = 'greedy+mem'
        self.algostat['MCalgo'] = algorithm
        self.algostat['k'] = k
        self.algostat['result']['edges'] = E
        # self.Query.eval()
        self.algostat['result']['H*'] = self.algostat['result']['H0']-history[-1]
        self.algostat['execution_time'] = e_tm
        # self.algostat['support'] = str(list(self.Query.phiInv.keys()))
        self.algostat['DeltaH'] = self.algostat['result']['H0'] - self.algostat['result']['H*']
        self.algostat['|DeltaH|'] = abs(self.algostat['result']['H0'] - self.algostat['result']['H*'])
        self.algostat['history_deltaH'] = history
        if self.debug: self.algostat['M'] = M
        return E,self.algostat['result']['H*'], self.algostat['result']['H0']- self.algostat['result']['H*']
    
    def algorithm6(self, property, algorithm, k, K, update_type = 'o1',verbose = False):
        """ Variant of Algorithm 5 where CT is approximated via sampling"""
        if verbose: print('Query eval.')
        self.algostat['result']['H0'] = self.measure_H0(property,algorithm,K)
        start_execution_time = time()
        # Start of Algorithm 5
        self.Query.constructTables_S(K = K, verbose = verbose) 
        Estar = copy(self.G.edict)
        E = []
        if (verbose):
            print('H0: ',self.algostat['result']['H0'])
            print('p(e): ',self.G.edict)
            print('Pr[G]: ',self.Query.PrG)
            print('G: ',self.Query.G)
            # print('Pr[Omega]: ', self.Query.freq_distr)
            # print('results: ',self.Query.results)
        for iter in range(k):
            if (verbose):
                print("Iteration: ",iter)
                print('Pr[G]: ',self.Query.PrG)
                
            # Start of Algorithm 4
            
            # Construct Pr[Omega] 
            Pr_Omega = {}
            for omega in self.Query.phiInv:
                Pr_Omega[omega] = sum([self.Query.PrG[i] for i in self.Query.phiInv[omega]])

            # Calculate Pr_up^{e_j}(G_i) for all i,j
            self.compute_Pr_up(verbose = verbose)
            
            if (verbose):
                # self.Query.eval()
                H0 = self.measure_H0(property,algorithm,K)
                # print('H0: ',H0)

            if (verbose):
                print('Selecting edge#: ',iter+1)
                print('Pr[Omega]: ',Pr_Omega)
                print("Pr_up[Omega]: ",self.Pr_up)
                # print('Gsums: ',self.Gsums)
            local_maxima = None 
            for e in Estar: # Among remaining edges (Estar), find the one (e*) with largest  H - H_{up}
                DeltaHe2 = 0 # H - H^{e}_{up}
                for omega in self.Query.phiInv:
                    if (Pr_Omega[omega] != 0):
                        if (self.Pr_up[omega][e] != 0):
                            DeltaHe2 += (log2(self.Pr_up[omega][e])*self.Pr_up[omega][e] - log2(Pr_Omega[omega])*Pr_Omega[omega])
                        else:
                            DeltaHe2 += (0 - log2(Pr_Omega[omega])*Pr_Omega[omega])
                    else:
                        if (self.Pr_up[omega][e] != 0):
                            DeltaHe2 += (log2(self.Pr_up[omega][e])*self.Pr_up[omega][e] - 0)
                        else:
                            DeltaHe2 += 0

                if verbose:
                    print('e: ',e,' DeltaH2[e]: ',DeltaHe2)
                    print('H_next [',e,']: ', H0 + DeltaHe2)
                
                if local_maxima is not None:
                    if DeltaHe2 > local_maxima[1]:
                        local_maxima = (e,DeltaHe2)
                else:
                   local_maxima = (e,DeltaHe2)
        
            estar = local_maxima[0] # assign e*
            print('e: ',local_maxima[0],' DeltaH2[e]: ',local_maxima[1])
            # End of Algorithm 4
            E.append(estar)
            if (verbose):
                print('e* = ',estar)
            del Estar[estar] # Delete e* from dictionary s.t. next iteration is 1 less than the current.
            self.G.edge_update(estar[0],estar[1],type = update_type) # Update UGraph()
            self.Query.reset(self.G) # Re-initialise Query() with updated UGraph()
            # Instead of updateing the table we construct from scratch by re-sampling
            if iter < k-1:
                self.Query.constructTables_S(K = K, verbose = verbose) 

            if (verbose):
                print('After p(e) update: ')
                # print('C = ',self.Query.C)
            if (verbose):
                print('----------')
        self.algostat['algorithm'] = 'greedy+mem+resamp'
        self.algostat['MCalgo'] = algorithm
        self.algostat['k'] = k
        self.algostat['result']['edges'] = E
        self.algostat['result']['H*'] = self.measure_H0(property,algorithm,K)
        self.algostat['execution_time'] = time() - start_execution_time
        # self.algostat['support'] = str(list(self.Query.phiInv.keys()))
        self.algostat['DeltaH'] = self.algostat['result']['H0'] - self.algostat['result']['H*']
        self.algostat['|DeltaH|'] = abs(self.algostat['result']['H0'] - self.algostat['result']['H*'])
        return E,self.algostat['result']['H*'],self.algostat['result']['H0'] - self.algostat['result']['H*']
    
        # Greedy k-edge selection: Algorithm 3

    def greedy(self,  property, algorithm, k,  N = 1, T = 1, update_type = 'o1', verbose = False):
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
        H0 = self.measure_H0(property,algorithm,T, N)
        self.algostat['result']['H0'] = H0
        E = []
        if (verbose):
            print('H0: ',H0)
            print('p(e): ',self.G.edict)
            print('Pr[Omega]: ', self.Query.freq_distr)
            # print('results: ',self.Query.results)
        for i in range(k):
            # print('iteration: ',i)
            local_maxima = None 
            if (verbose):
                print('Selecting edge#: ',i+1)
            # Estar= [Ecand[i]]
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
                self.Query.eval()
                if (verbose):
                    print('considering e= ',e)
                    # print('p(e): ',g_copy.edict)
                    print('Pr[Omega]: ', self.Query.freq_distr)
                    # print('results: ',self.Query.results)
                Hi = self.Query.compute_entropy()
                if local_maxima:
                    if H0 -  Hi > H0 - local_maxima[1]:
                        local_maxima = (e, Hi)
                        if (verbose):
                            print('new maxima: ',local_maxima)
                else:
                    local_maxima = (e, Hi)
                    if (verbose):
                        print('initial maxima: ',local_maxima)
                # print('e: ',e,' \Delta H = ', H0-Hi, ' H0:', H0, ' H_next: ',Hi)
            estar = local_maxima[0] # assign e*
            H0 = local_maxima[1] # Assign H0 for the next iteration.
            E.append(estar)
            if (verbose):
                print('e* = ',estar)
            del Estar[estar] # Delete e* from dictionary s.t. next iteration is 1 less than the current.
            self.G.edge_update(estar[0],estar[1],type = update_type) # Update UGraph()
            self.Query.reset(self.G) # Re-initialise Query() with updated UGraph()
            history.append(self.algostat['result']['H0']-self.measure_H0(property,algorithm,T,N))
            if (verbose):
                self.Query.eval()
                print('Entropy of Updated UGraph: ',self.Query.compute_entropy())

        self.algostat['execution_time'] = time() - start_execution_time
        self.algostat['MCalgo'] = algorithm
        self.algostat['result']['edges'] = E
        self.algostat['result']['H*'] = H0
        self.algostat['support'] = ''
        self.algostat['DeltaH'] = self.algostat['result']['H0'] - self.algostat['result']['H*']
        self.algostat['|DeltaH|'] = abs(self.algostat['result']['H0'] - self.algostat['result']['H*'])
        self.algostat['history_deltaH'] = history
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

    def crowd_kbest(self, property, algorithm, k, K, update_dict, N=1,T=1, update_type = 'c1', verbose = False):
        """ Variant of Algorithm 5 where CT is approximated via sampling"""
        print(['adaptive','non-adaptive'][update_type == 'c1'], ' setting')
        assert k>=1
        history = []
        start_execution_time = time()
        if verbose: print('Query eval.')
        self.algostat['result']['H0'] = self.measure_H0(property,algorithm,T,N)
        # Start of Algorithm 5
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
        for iter in range(k):
            if (verbose):
                print("Iteration: ",iter)
            # Start of Algorithm 4
            # print('Pr(G_i): ', self.Query.PrG)
            # Construct Pr[Omega] 
            Pr_Omega = {}
            for omega in self.Query.phiInv:
                Pr_Omega[omega] = sum([self.Query.PrG[i] for i in self.Query.phiInv[omega]])
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
            local_maxima = None 
            for e in Estar: # Among remaining edges (Estar), find the one (e*) with largest  H - H_{up}
                DeltaHe2 = 0 # H - H^{e}_{up}
                H_up = 0
                p_e = self.G.edict[e]
                for omega in self.Query.phiInv:
                    if Pr_Omega[omega] == 0:
                        h0 = 0
                    else:
                        h0 = log2(Pr_Omega[omega])*Pr_Omega[omega]

                    if self.Pr_up[omega].get(e,0) == 0:
                        h1 = 0
                    else:
                        h1 = p_e*log2(self.Pr_up[omega][e])*self.Pr_up[omega][e]
                    if self.Pr_up_0[omega].get(e,0) == 0:
                        h2 = 0
                    else:
                        h2 = (1-p_e)*log2(self.Pr_up_0[omega][e])*self.Pr_up_0[omega][e]
                    expected_entropy = h1 + h2
                    DeltaHe2 += (expected_entropy - h0)
                    H_up += -expected_entropy
                    
                # if verbose:
                if self.debug:
                    print('H_up^e', e,' -> ',H_up)
                    print('e: ',e,' DeltaH2[e]: ',DeltaHe2)
                    # print('H_next [',e,']: ', self.algostat['result']['H0'] + DeltaHe2)
                
                if local_maxima is not None:
                    if DeltaHe2 > local_maxima[1]:
                        local_maxima = (e,DeltaHe2)
                else:
                   local_maxima = (e,DeltaHe2)
            if verbose: print('e*: ',local_maxima[0],' DeltaH2[e]: ',local_maxima[1])
            estar = local_maxima[0] # assign e*
            # End of Algorithm 4
            E.append(estar)
            del Estar[estar] # Delete e* from dictionary s.t. next iteration is 1 less than the current.
            if update_type == 'c2':
                if iter < k-1:
                    self.Query.adaptiveUpdateTables(estar,update_dict[estar], self.Pr_up,self.Pr_up_0)
                self.G.update_edge_prob(estar[0],estar[1],update_dict[estar])
                # print('-',self.Query.PrG)
                self.Query.reset(self.G) # Re-initialise Query() with updated UGraph()  
                history.append(self.algostat['result']['H0']-self.measure_H0(property,algorithm,T,N)) 
            
            if (verbose):
                print('----------')
        if update_type == 'c1':
            for e in E:
                self.G.update_edge_prob(estar[0],estar[1],update_dict[estar])
                self.Query.reset(self.G) # Re-initialise Query() with updated UGraph()  
                history.append(self.algostat['result']['H0']-self.measure_H0(property,algorithm,T,N))

        e_tm = time() - start_execution_time
        self.algostat['algorithm'] = 'greedy+mem'
        self.algostat['MCalgo'] = algorithm
        self.algostat['k'] = k
        self.algostat['result']['edges'] = E
        # self.Query.eval()
        self.algostat['result']['H*'] = self.algostat['result']['H0']-history[-1]
        self.algostat['execution_time'] = e_tm
        # self.algostat['support'] = str(list(self.Query.phiInv.keys()))
        self.algostat['DeltaH'] = self.algostat['result']['H0'] - self.algostat['result']['H*']
        self.algostat['|DeltaH|'] = abs(self.algostat['result']['H0'] - self.algostat['result']['H*'])
        self.algostat['history_deltaH'] = history
        if self.debug: self.algostat['M'] = M
        return E,self.algostat['result']['H*'], self.algostat['result']['H0']- self.algostat['result']['H*']
