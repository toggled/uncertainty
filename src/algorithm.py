from src.graph import UGraph as Graph 
from copy import deepcopy,copy
import random 
from src.query import Query
from time import time 
import math 
from math import log2
import tracemalloc

class Algorithm:
    """ A generic Algorithm class that implmenets all the Exact algorithms in the paper."""
    def __init__(self, g, query) -> None:
        self.algostat = {} 
        self.G = deepcopy(g)
        assert isinstance(self.G, Graph)
        assert isinstance(query, Query)
        self.Query = query
        self.algostat['execution_time'] = 0
        self.algostat['result'] = {}
        self.algostat['k'] = 0
        self.algostat['algorithm'] = ''
        self.algostat['support'] = []
        self.algostat['peak_memB'] = []

    def measure_uncertainty(self):
        """ 
        Measures entropy exactly by sampling all possible worlds. 
        """
        memory_consumption_exact = []
        tracemalloc.start()
        start_execution_time = time()
        self.Query.eval(dontenumerateworlds = True)
        tm2 = time()
        H = self.Query.compute_entropy()
        current_mem_exact, peak_mem_exact = tracemalloc.get_traced_memory()
        memory_consumption_exact.append(peak_mem_exact)
        self.algostat['execution_time'] = time() - start_execution_time
        self.algostat['algorithm'] = 'exact'
        self.algostat['H'] = H 
        self.algostat['support'] =  str(list(self.Query.get_support()))
        self.algostat['query_eval_tm'] = sum(self.Query.evaluation_times) # time spent on query evlauation
        self.algostat['total_sample_tm'] = (tm2-start_execution_time) - self.algostat['query_eval_tm'] # Time spend on possible world sampling
        self.algostat['peak_memB'] = max(memory_consumption_exact)
        return H
         

    # Exhaustive search algorithm. All (m choose k) choices are explored. 
    def Bruteforce(self, k, update_type = 'o1',verbose = False):
        """ 
        Brute-force algorithm
        Returns selected edgeset, entropy value after selection, and entropy-reduction amount (DeltaH)
        """
        assert k>=1
        self.algostat['algorithm'] = 'Bruteforce'
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

            if self.Global_maxima:
                if H0 -  Hi > H0 - self.Global_maxima[1]:
                    self.Global_maxima = (edge_set, Hi)
                    if (verbose):
                        print('new maxima: ',self.Global_maxima)
            else:
                self.Global_maxima = (edge_set, Hi)
                if (verbose):
                    print('Initial maxima: ',self.Global_maxima)
        self.algostat['execution_time'] = time() - start_execution_time
        self.algostat['result']['edges'] = self.Global_maxima[0]
        self.algostat['result']['H*'] = self.Global_maxima[1]
        self.algostat['support'] = ''
        return self.Global_maxima[0], self.Global_maxima[1],  H0 - self.Global_maxima[1]
        
    # Greedy k-edge selection: Algorithm 3
    def algorithm3(self,  k, update_type = 'o1',verbose = False):
        """
        Exact version of Algorithm 3
        Returns selected edgeset, entropy value after selection, and entropy-reduction amount (DeltaH)
        """
        assert k>=1
        self.algostat['algorithm'] = 'Alg3'
        self.algostat['k'] = k
        start_execution_time = time()
        Estar = copy(self.G.edict)
        self.Query.eval()
        H0 = self.Query.compute_entropy() # Initial entropy
        self.algostat['result']['H0'] = H0
        E = []
        if (verbose):
            print('H0: ',H0)
            print('p(e): ',self.G.edict)
            print('Pr[Omega]: ', self.Query.freq_distr)
            # print('results: ',self.Query.results)
        for i in range(k):
            local_maxima = None 
            if (verbose):
                print('Selecting edge#: ',i+1)
            for e in Estar: # Among remaining edges (Estar), find the one (e*) with largest reduction in Entropy
                g_copy = deepcopy(self.G)
                # if (verbose):
                #     print('g_copy: ',g_copy.edict)
                g_copy.edge_update(e[0],e[1], type= update_type)
                self.Query.reset(g_copy)
                self.Query.eval()
                if (verbose):
                    print('considering e= ',e)
                    print('p(e): ',g_copy.edict)
                    print('Pr[Omega]: ', self.Query.freq_distr)
                    print('results: ',self.Query.results)
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
            estar = local_maxima[0] # assign e*
            H0 = local_maxima[1] # Assign H0 for the next iteration.
            E.append(estar)
            if (verbose):
                print('e* = ',estar)
            del Estar[estar] # Delete e* from dictionary s.t. next iteration is 1 less than the current.
            self.G.edge_update(estar[0],estar[1],type = update_type) # Update UGraph()
            self.Query.reset(self.G) # Re-initialise Query() with updated UGraph()

            if (verbose):
                self.Query.eval()
                print('Entropy of Updated UGraph: ',self.Query.compute_entropy())

        self.algostat['execution_time'] = time() - start_execution_time
        self.algostat['result']['edges'] = E
        self.algostat['result']['H*'] = H0
        self.algostat['support'] = ''
        return E,self.algostat['result']['H*'],self.algostat['result']['H0']- self.algostat['result']['H*']

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

    def compute_Pr_up(self,op, verbose = False):
        """ 
        Calculate 
        Pr_up(G_i,e_j) (Pr_up(G_i) after p(e_j) is updated ) 
        """
        self.Pr_up = {}
        if (verbose):
            self.Gsums = {}
        for omega in self.Query.phiInv:
            self.Pr_up[omega] = {}
            if (verbose):
                self.Gsums[omega] = {}
            for e in self.G.Edges:
                p_e = self.G.edict[e] # p(e)
                p_up_e = self.G.get_next_prob(e[0],e[1],op) # p_up(e)
                _sumPrG = 0
                if (verbose):
                    self.Gsums[omega][e] = []
                for i in self.Query.phiInv[omega]:
                    if e in self.Query.index[i]: # if edge e is in G_i where omega is observed
                        if p_e != p_up_e:
                            G_up = self.Query.PrG[i] * p_up_e / p_e  # Pr_up(G_i) = Pr(G) * p_up(e) / p(e)
                            _sumPrG += G_up 
                            if (verbose):
                                self.Gsums[omega][e].append((i,1))
                    else: #  edge e not in G_i where omega is observed
                        if p_e != p_up_e:
                            try:
                                G_up = self.Query.PrG[i] * (1-p_up_e) / (1-p_e)  # Pr_up(G_i) = Pr(G_i) * (1- p_up(e)) / (1- p(e))
                                _sumPrG += G_up 
                                if (verbose):
                                    self.Gsums[omega][e].append((i,0))
                            except ZeroDivisionError:
                                print(p_e,p_up_e,self.Query.PrG[i])
                                raise ZeroDivisionError
                                
                self.Pr_up[omega][e] = _sumPrG

    def algorithm5(self, k, update_type = 'o1',verbose = False):
        """ 
        Algorithm 5 (with Exact memoization)
        Returns selected edgeset, entropy value after selection, and entropy-reduction amount (DeltaH)
        """
        self.algostat['algorithm'] = 'Alg5'
        self.algostat['k'] = k
        self.Query.eval()
        self.algostat['result']['H0'] =  self.Query.compute_entropy() # Initial entropy. Kept outside of time() because H0 is not needed in contr table and computed only for logging.
        start_execution_time = time()
        self.Query.constructTables(op = update_type,verbose = verbose) 
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
            # Start of Algorithm 4
            
            # Construct Pr[Omega] 
            Pr_Omega = {}
            for omega in self.Query.phiInv:
                Pr_Omega[omega] = sum([self.Query.PrG[i] for i in self.Query.phiInv[omega]])

            # Calculate Pr_up^{e_j}(G_i) for all i,j
            self.compute_Pr_up(update_type,verbose = verbose)
            
            if (verbose):
                self.Query.eval()
                H0 = self.Query.compute_entropy()
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
                for omega in self.Query.phiInv:
                    if (Pr_Omega[omega] != 0):
                        if (self.Pr_up[omega][e] != 0):
                            DeltaHe2 += (log2(self.Pr_up[omega][e],2)*self.Pr_up[omega][e] - log2(Pr_Omega[omega],2)*Pr_Omega[omega])
                        else:
                            DeltaHe2 += (0 - log2(Pr_Omega[omega],2)*Pr_Omega[omega])
                    else:
                        if (self.Pr_up[omega][e] != 0):
                            DeltaHe2 += (log2(self.Pr_up[omega][e],2)*self.Pr_up[omega][e] - 0)
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
            # End of Algorithm 4
            E.append(estar)
            if (verbose):
                print('e* = ',estar)
            del Estar[estar] # Delete e* from dictionary s.t. next iteration is 1 less than the current.
            self.Query.updateTables(estar, update_type)
            self.G.edge_update(estar[0],estar[1],type = update_type) # Update UGraph()
            self.Query.reset(self.G) # Re-initialise Query() with updated UGraph()
            
            if (verbose):
                print('After p(e) update: ')
                # print('C = ',self.Query.C)
            if (verbose):
                print('----------')
        self.algostat['execution_time'] = time() - start_execution_time
        self.algostat['result']['edges'] = E
        self.Query.eval()
        self.algostat['result']['H*'] = self.Query.compute_entropy()
        # self.algostat['support'] = ','.join([str(i) for i in self.Query.phiInv.keys()])
        # self.algostat['support'] = str(list(self.Query.phiInv.keys()))
        self.algostat['support'] = ''
        return E,self.algostat['result']['H*'],self.algostat['result']['H0']- self.algostat['result']['H*']

class ApproximateAlgorithm:
    """ Implements Algorithm 2 and Approximate variants of Algorithms 3, and 5"""
    def __init__(self, g, query) -> None:
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
        self.algostat['peak_memB'] = []
    
    def measure_uncertainty(self, N=1, T=10):
        """
        Alg 2 
        """
        memory_consumption_appr = []
        tracemalloc.start()
        start_execution_time = time()
        query_evaluation_times = []
        hat_H_list = []
        support_observed = set()
        sum_H = 0
        for i in range(N):
            hat_Pr = {}
            for g in self.G.get_Ksample(T,seed=i):
                # if i==0:
                # print(g[0])
                start_tm = time()
                omega = self.Query.evalG(g[0])
                query_evaluation_times.append(time()-start_tm)
                hat_Pr[omega] = hat_Pr.get(omega,0) + 1.0/T
                support_observed.add(omega)
            hat_H = -sum([hat_Pr[omega] * log2(hat_Pr[omega]) for omega in hat_Pr])
            hat_H_list.append(hat_H)
            sum_H += hat_H 
        mean_H =  sum_H /N
        current_mem_appr, peak_mem_appr = tracemalloc.get_traced_memory()
        memory_consumption_appr.append(peak_mem_appr)
        self.algostat['execution_time'] = time() - start_execution_time
        self.algostat['algorithm'] = 'appr'
        self.algostat['H'] = mean_H
        self.algostat['support'] =  str(list(support_observed))
        self.algostat['total_sample_tm'] = sum(self.G.sample_time_list)
        self.algostat['query_eval_tm'] = sum(query_evaluation_times)
        self.algostat['peak_memB'] = max(memory_consumption_appr)
        return mean_H 

    def algorithm3(self,  k, update_type = 'o1',verbose = False):
        """ Variant of Algorithm 3 where entropy is approximated via sampling, not exactly computed """
        # TODO: Sampler code: Ugraph().get_Ksample().
        self.algostat['algorithm'] = 'Alg3'
        self.algostat['k'] = k
        pass 

    def compute_Pr_up(self,op, verbose = False):
        """ 
        Calculate 
        Pr_up(G_i,e_j) (Pr_up(G_i) after p(e_j) is updated ) 
        """
        self.Pr_up = {}
        if (verbose):
            self.Gsums = {}
        for omega in self.Query.phiInv:
            self.Pr_up[omega] = {}
            if (verbose):
                self.Gsums[omega] = {}
            for e in self.G.Edges:
                p_e = self.G.edict[e] # p(e)
                p_up_e = self.G.get_next_prob(e[0],e[1],op) # p_up(e)
                _sumPrG = 0
                if (verbose):
                    self.Gsums[omega][e] = []
                for i in self.Query.phiInv[omega]:
                    if e in self.Query.index[i]: # if edge e is in G_i where omega is observed
                        if p_e != p_up_e:
                            G_up = self.Query.PrG[i] * p_up_e / p_e  # Pr_up(G_i) = Pr(G) * p_up(e) / p(e)
                            _sumPrG += G_up 
                            if (verbose):
                                self.Gsums[omega][e].append((i,1))
                    else: #  edge e not in G_i where omega is observed
                        if p_e != p_up_e:
                            try:
                                G_up = self.Query.PrG[i] * (1-p_up_e) / (1-p_e)  # Pr_up(G_i) = Pr(G_i) * (1- p_up(e)) / (1- p(e))
                                _sumPrG += G_up 
                                if (verbose):
                                    self.Gsums[omega][e].append((i,0))
                            except ZeroDivisionError:
                                print(p_e,p_up_e,self.Query.PrG[i])
                                raise ZeroDivisionError
                                
                self.Pr_up[omega][e] = _sumPrG

    def algorithm5(self, k, K, update_type = 'o1',verbose = False):
        """ Variant of Algorithm 5 where CT is approximated via sampling"""
        # TODO: Sampler code: Ugraph().get_Ksample().
        self.algostat['algorithm'] = 'Alg5-S'
        self.algostat['k'] = k
        self.Query.eval()
        self.algostat['result']['H0'] =  self.Query.compute_entropy() # Initial entropy. Kept outside of time() because H0 is not needed in contr table and computed only for logging.
        start_execution_time = time()
        # Start of Algorithm 5
        self.Query.constructTables_S(op = update_type, K = K, verbose = verbose) 
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
            # Start of Algorithm 4
            
            # Construct Pr[Omega] 
            Pr_Omega = {}
            for omega in self.Query.phiInv:
                Pr_Omega[omega] = sum([self.Query.PrG[i] for i in self.Query.phiInv[omega]])

            # Calculate Pr_up^{e_j}(G_i) for all i,j
            self.compute_Pr_up(update_type,verbose = verbose)
            
            if (verbose):
                self.Query.eval()
                H0 = self.Query.compute_entropy()
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
                for omega in self.Query.phiInv:
                    if (Pr_Omega[omega] != 0):
                        if (self.Pr_up[omega][e] != 0):
                            DeltaHe2 += (log2(self.Pr_up[omega][e],2)*self.Pr_up[omega][e] - log2(Pr_Omega[omega],2)*Pr_Omega[omega])
                        else:
                            DeltaHe2 += (0 - log2(Pr_Omega[omega],2)*Pr_Omega[omega])
                    else:
                        if (self.Pr_up[omega][e] != 0):
                            DeltaHe2 += (log2(self.Pr_up[omega][e],2)*self.Pr_up[omega][e] - 0)
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
            # End of Algorithm 4
            E.append(estar)
            if (verbose):
                print('e* = ',estar)
            del Estar[estar] # Delete e* from dictionary s.t. next iteration is 1 less than the current.
            self.Query.updateTables(estar, update_type)
            self.G.edge_update(estar[0],estar[1],type = update_type) # Update UGraph()
            self.Query.reset(self.G) # Re-initialise Query() with updated UGraph()
            
            if (verbose):
                print('After p(e) update: ')
                # print('C = ',self.Query.C)
            if (verbose):
                print('----------')
        self.algostat['execution_time'] = time() - start_execution_time
        self.algostat['result']['edges'] = E
        self.Query.eval()
        self.algostat['result']['H*'] = self.Query.compute_entropy()
        self.algostat['support'] = str(list(self.Query.phiInv.keys()))
        return E,self.algostat['result']['H*'],self.algostat['result']['H0']- self.algostat['result']['H*']
 