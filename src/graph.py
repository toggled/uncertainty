import itertools 
import networkx as nx
from matplotlib import pyplot as plt 
import random 
from time import time 
from copy import deepcopy
from heapq import heappush, heappop
from networkx.algorithms.bipartite.matching import INFINITY
from math import sqrt,ceil,floor,log
from collections import deque 
class UGraph:
    """
    A generic class for Uncertain Graph data structure and for various operations on it. 
    Supports edge weights
    """
    def __init__(self):
        self.Edges = [] # Edge list of the uncertain graph
        self.edict = {} # Edge -> Prob mapping of the uncertain graph.
        # self.notedict = {} # Edge -> (1-Prob) mapping of the uncertain graph. Mostly to assist faster computation.
        # self.sample_time_list = [] # exec. time to generate individual possible worlds
        self.total_sample_tm = 0
        self.weights = {}
        self.nbrs = {} # neighborlist representation.
        self.nx_format = nx.Graph()

    def clear(self):
        self.Edges = []
        self.edict = {}
        # self.notedict = {}
        self.weights = {}
        del self.nbrs 
        del self.nx_format
        
    def __str__(self):
        return ' '.join(self.Edges)
    
    def __len__(self):
        """ Returns number of edges """
        return len(self.Edges)
    
    def get_prob(self, e):
        u,v = min(e),max(e) 
        return self.edict[(u,v)]

    def count_nodes(self):
        if len(self.nbrs) == 0:
            d = set()
            for e in self.Edges:
                if e[0] not in d:
                    d.add(e[0])
                if e[1] not in d:
                    d.add(e[1])
            return len(d)
        else:
            return len(self.nbrs)
    
    def construct_nbrs(self):
        if len(self.nbrs) == 0:
            nbrs = {}
            for e in self.Edges:
                _tmp = nbrs.get(e[0],[])
                _tmp.append(e[1])
                nbrs[e[0]] = _tmp 
                _tmp = nbrs.get(e[1],[])
                _tmp.append(e[0])
                nbrs[e[1]] = _tmp 
            self.nbrs = nbrs
        else:
            nbrs = self.nbrs
        return nbrs 
    
    def count_tri_approx(self, sample_edges):
        num_nodes = 0 # total number of nodes in the possible worlds
        nodeset = set() # set of nodes with deg >=2
        nbrs = {} # #nbrs for all the nodes in the subgraph induced by edges in sample_edges.
        # This loop computes (1) #nbrs for all nodes (2) constructs nodeset and (3) count num nodes
        for e in sample_edges: 
            # print(e)
            u,v = e[0],e[1]
            _tmp = nbrs.get(u,[])
            num_nbrs_u = len(_tmp)
            if (num_nbrs_u==0):
                num_nodes += 1 # If first time inserted into dic nbrs. count it.
            else:
                if (num_nbrs_u+1)>=2:
                    nodeset.add(u)
            _tmp.append(v)
            nbrs[u] = _tmp 

            _tmp = nbrs.get(v,[])
            num_nbrs_v = len(_tmp)
            if (num_nbrs_v==0):
                num_nodes += 1 # If first time inserted into dic nbrs. count it.
            else:
                if (num_nbrs_v+1)>=2:
                    nodeset.add(v)
            _tmp.append(u)
            nbrs[v] = _tmp 
        # print(nbrs)
        # Construct parameter value for approximation algorithm 
        nu = 100 # 1000 # prob of having good estimate is at least 99%
        eps = 1/sqrt(num_nodes) # +-sqrt(n) error will be incurred during tri counting , but with prob at most 1 - ((nu -1)/nu)
        k = ceil(log(2*nu)/(2*eps**2))
        # print('approximate triangle counting: nu = ',nu,' eps = ',eps,' n = ',num_nodes, ' k = ',k)

        V = list(nodeset) # set of nodes whose deg >=2
        absV = len(V)
        # print('deg > 2 = ',absV)
        # approximate counting
        num_tri = 0
        for i in range(k):
            j = random.randint(0,absV-1)
            nbrs_u = nbrs[V[j]]
            v,w = random.sample(nbrs_u,k=2)
            if w in nbrs[v]:
                num_tri += 1
        return floor(absV * (num_tri/k))
    
    def find_rel_rss(self,T, source,target,seed = 1, optimiseargs = {'nbrs':None, 'doopt': False}):
        print('RSS: source: ',source, ' target: ',target)
        kRecursiveSamplingThreshold = 30
        r = 2
        time_spent = {'mc': 0,'genT':0}
        verbose = False # True 
        def generate_bit_vectors(n, bit_vector=''):
            if n == 0:
                yield bit_vector
            else:
                yield from generate_bit_vectors(n - 1, bit_vector + '0')
                yield from generate_bit_vectors(n - 1, bit_vector + '1')

        # states = [x for x in generate_bit_vectors(r)]
        states = []
        
        for i in range(r+1):
            if i == 0:
                states.append(''.join(['0']*r))
                continue
            x = ['0']*r 
            x[i-1] = '1'
            states.append(str(''.join(x)))
        # print(states)
        # states = ["0001","0010","0011","0100","0101",\
        #           "0110","0111","1000","1001","1010",\
        #          "1011","1100","1101","1110","1111",\
        #         "0000"]
        start_execution_time = time()
        assert len(self.nbrs)>0
        if optimiseargs is not None:
            if optimiseargs['nbrs'] is None:
                nbrs = self.construct_nbrs() # Constructs node => Nbr incidence dictionary. 
            else:
                nbrs = optimiseargs['nbrs']
        else:
            if verbose: print('constructing nbrs.')
            nbrs = self.construct_nbrs()
        if (source not in nbrs) or (target not in nbrs):
            print('source, target pair not exists in the graph')
            print('len(nbrs) = ',len(nbrs))
            return 0.0,nbrs
        # print(nbrs)
        # sv_map = set([source])
        edge_map = {}
        edges = [] # contains bfs traversal ordering of edges from source
        # si_queue = [(source,t) for t in set(nbrs[source])]
        temp_q = [source]
        while len(temp_q):
            v = temp_q.pop(0)
            nbrs_v = list(set(nbrs[v]))
            if len(nbrs_v):
                for w in nbrs_v:
                    vv,ww = min(v,w),max(v,w)
                    if (vv,ww) not in edge_map:
                        edge_map[(vv,ww)] = True 
                        temp_q.append(w)
                        edges.append((v,w))
        # print('edges = ',edges,' ',self.Edges)
        # verbose = False
        # assert len(si_queue) >= r
        # sv_map = set([source])
        #print(self.Edges)
        # if verbose: print('si_q: ',edges[:r], ': nbrs[source]: ',set(nbrs[source]))
        # print('seed = ',seed)
        # random.seed(seed)
        # random.seed(int(time()))
        def samplingR_RSS(si_queue, forbidden_edges):
            r = random.randint(1,100) + seed 
            ss = int(time()%r)
            # print('seed = ',ss)
            random.seed(ss)
            # verbose = False 
            # print('------sampline RSS------ ')
            # sv_map => nodes visited already.
            # si_queue => initially it holds incident edges on source.
            poss_world = []
            # Returns: 1/0 if the sampled possible world contains (does not) target node 
            # temp_q = [] # If edges e1,e2,.. in si_queue is selected, it holds [e1[1],e2[1],..]
            sv_map = deque([source])
            # print('si_queue = ',si_queue)
            # temp_q = si_queue
            # while len(si_queue): 
            #     e = si_queue.pop(0)
            #     p = self.get_prob(e)
            #     # if e[1] in sv_map:
            #     #     continue 
            #     if random.random() < p:
            #         poss_world.append(e)
            #         if e[1] == target:
            #             return 1 
            #         sv_map.add(e[1])
            #         sv_map.add(e[0])
            #         temp_q.append(e[1])
            # # si_queue is empty at this stage. sv_map, temp_q contains nodes from selected edges in si_queue.
            # print('temp_q= ',temp_q)
            visited = {}
            while len(sv_map):
                # print('sv_map: ',len(sv_map))
                v = sv_map.popleft()
                visited[v] = True
                
                nbrs_v = list(set(nbrs[v]))
                if verbose: print('visiting v = ',v, ' nbrs_v: ',nbrs_v, ' visited: ', visited)
                if len(nbrs_v):
                    for w in nbrs_v:
                        if w in visited:
                            if verbose: print('(',v,w,')', 'ignored because w visited already.')
                            continue 
                        # 3 types of edges: Forbidden (0), always exists(1)
                        # and uncertaint (*)
                        if (v,w) in forbidden_edges or (w,v) in forbidden_edges:
                            if verbose: print('(',v,w,')', 'ignored because it is forbidden.')
                            # print('cannot visit forbidden edge : ',(v,w))
                            # visited[w] = True 
                            continue 
                        elif (v,w) in si_queue or (w,v) in si_queue:
                            # visited[w] = True 
                            if w==target:
                                if verbose: poss_world.append((v,w))
                                return 1
                            if w not in visited:
                                # print('will visit nbr -> ',w)
                                sv_map.append(w)
                                if verbose: poss_world.append((v,w))
                            continue 
                        else:
                            p_vw = self.get_prob((v,w))
                            r = random.random()
                            if verbose: print('coin toss. ',r,p_vw, 'w = ',w)
                            if r < p_vw:
                            # if r >= p_vw:
                                # print('r< p_vw')
                                if w == target:
                                    if verbose: print('target ',target,' found')
                                    if verbose: poss_world.append((v,w))
                                    if verbose: print('world: ',poss_world)
                                    return 1 
                                if w not in visited:
                                    # print('will visit nbr -> ',w)
                                    sv_map.append(w)
                                    # temp_q.append(w)
                                    if verbose: poss_world.append((v,w))
                            else:
                                if verbose: print('(',v,w,')', 'ignored because coin toss prob too low.')
            if verbose: print('world: ',poss_world, 'visited = ',visited)
            return 0 
        
        def findReliability_RHH_forRSS(sv_map, si_queue, n, flag, node):
            # print('->', sv_map,si_queue, n, flag, node)
            if flag:
                if node == target:
                    return 1 
                sv_map.add(node)
                si_queue += [(node,t) for t in set(nbrs[node])]
            if len(si_queue)==0:
                return 0
            # print('si_queue before: ',len(si_queue))
            if len(si_queue)>4:
                return findReliability_RSS(deepcopy(sv_map),deepcopy(si_queue),n,15,[])
            # print('si_queu aftre: ',len(si_queue))
            if n <= kRecursiveSamplingThreshold:
                if n == 0:
                    return 0 
                rhh = 0
                for i in range(n):
                    r = samplingR_RSS(deepcopy(sv_map),deepcopy(si_queue))
                    # print(i,' : ',r)
                    rhh += r
                print('rhh for rss: ',rhh*1.0/n)
                return rhh*1.0/n
            e = si_queue.pop(0)
            while (e[1] in sv_map):
                if len(si_queue) == 0:
                    return 0 
                e = si_queue.pop(0)
            # print('->',e)
            pe = self.get_prob(e)
            # print(' ->-> ', math.floor(n*pe), n- math.floor(n*pe))
            return pe * findReliability_RHH_forRSS(deepcopy(sv_map),deepcopy(si_queue),floor(n*pe),True,e[1]) + \
            (1- pe)* findReliability_RHH_forRSS(deepcopy(sv_map),deepcopy(si_queue), n - floor(n*pe),False,e[1]) 
        
        def findReliability_RSS(si_queue, forbidden, n, flag, r,time_spent={}): 
            # num_stratums = int(2**r) - 1
            num_stratums = r
            # forbidden = []
            # for i in range(r):
            #     if (states[flag][i] == '0'):
            #         forbidden.append(edges[i])
            if verbose: print('n = ',n,' flag = ',states[flag],' forbidden: ',len(forbidden), 'si_queue = ',si_queue)
        
            num_residual_edges = len(edges) - len(forbidden) 
            if verbose: print('#edges: ',len(edges),' #residual edges: ',num_residual_edges)            
            # if len(si_queue)==0:
            #     return 0
            # print('temp_q : ', temp_q)
            # print('si_queue 1: ',si_queue)
            if n <= kRecursiveSamplingThreshold or num_residual_edges< r:
                if verbose: print('base case: because n<=Threshold: ',n<=kRecursiveSamplingThreshold)
                if (n <= 0):
                    if verbose: print('(n<=0) => return 0: ',0)
                    return 0
                rhh = 0
                if verbose: print(states[flag])
                for i in range(n): #Run n times
                    s_tm = time()
                    rhh += samplingR_RSS(si_queue,forbidden)
                    time_spent['mc'] += (time() - s_tm)
                if verbose: print('MC sampling: ',rhh*1.0/n)
                return rhh*1.0 / n
            # Construct T / Select r edges from current UG (excluding forbidden edges) by running BFS from source
            if verbose: print('forbidden: ',forbidden,' si_queue: ',si_queue)
            s_tm = time()
            temp_q = deepcopy(si_queue)
            si_queue = []
            if len(si_queue)<r:
                for e in edges:
                    if (e not in forbidden) and (e not in temp_q):
                        si_queue.append(e)
                        if len(si_queue) == r:
                            break 
            time_spent['genT'] += (time() - s_tm)
            if verbose: print('T = ',si_queue)
            # nodes = []
            # edges= [] 
            # while len(nodes) < 4:
            #     print(len(nodes),'==',len(si_queue))
            #     if len(si_queue) == 0:
            #         si_queue += [e for e in edges]
            #         # print(si_queue, sv_map, nodes,target)
            #         print('findReliability_RHH_forRSS')
            #         return findReliability_RHH_forRSS(deepcopy(sv_map), deepcopy(si_queue), n, False, target)
            #         # return samplingR_RSS(deepcopy(sv_map), deepcopy(si_queue))
            #     e = si_queue.pop(0)
            #     if e[1] in sv_map:
            #         continue 
            #     else:
            #         # nodes.append(e[1])
            #         edges.append(e)

            # print('flag = ',flag,' edges: ',edges)
            reliablity= 0
            # temp_n = n 
            # temp_prob = 1.0 
            probs = []
            for i in range(r):
                probs.append(self.get_prob(si_queue[i]))
            for i in range(num_stratums+1):
                # if i == 15:
                #     reliablity += temp_prob * findReliability_RSS(si_queue, temp_n, 15)
                #     break 
                prob = 1.0
                temp_q2 = deepcopy(temp_q)
                # temp_q2 = []
                forbidden_i = deepcopy(forbidden)
                switch = False 
                for j in range(r):
                    if states[i][j] == '1':
                        prob = prob * probs[j]
                        temp_q2.append(si_queue[j])
                        switch = True 
                        break
                    else:
                        prob = prob * (1-probs[j])
                        if switch is False:
                            # forbidden_i.append(si_queue[j])
                            forbidden_i.add(si_queue[j])
                this_n = floor(n*prob)
                # temp_n -= this_n 
                # temp_prob -= prob 
                # if verbose: print('this_n = ',this_n)
                if verbose:
                    print('entering recursion: ',states[i])
                    print(si_queue, temp_q2, forbidden_i)
                
                if len(time_spent):
                    rel = findReliability_RSS(temp_q2,forbidden_i, this_n, i, r,time_spent)
                else:
                    rel = findReliability_RSS(temp_q2,forbidden_i, this_n, i, r)
                # print('---- ',i,flag,' -> ', rel)
                pi_i_mu_i = prob * rel
                reliablity += (pi_i_mu_i) 
            return reliablity
    
        
        probs = []
        for i in range(r):
            probs.append(self.get_prob(edges[i]))
        ui = []
        pi = []
        # Implicitely T <- the first r edges in edges list 
        # for i in range(int(2**r)):
        s_tm = time()
        for i in range(r+1):
            switch = False 
            # random.seed(i*int(time()))
            prob = 1.0
            temp_q = [] # Edges must exists.
            # forbidden = [] # Edges must not exists.
            forbidden = set()
            for j in range(r):
                if states[i][j] == '1':
                    switch = True 
                    prob = prob * probs[j]
                    temp_q.append(edges[j])
                    break
                else:
                    prob = prob * (1-probs[j])
                    if switch is False:
                        # forbidden.append(edges[j])
                        forbidden.add(edges[j])
            if verbose:
                print('Root: n = ',T,' flag = ',i, ' r = ',r, ' => ',states[i])
                print('------------')
            rl =  findReliability_RSS(temp_q, forbidden, n = floor(T*prob),flag = i, r = r,time_spent=time_spent)
            if verbose: 
                print('Root: returned from states ',states[i], 'reliability: ',rl)
                print('------------')
            ui.append(rl)
            pi.append(prob)
        e_tm = time() - s_tm 
        print('total time: ',e_tm, ' % spent on MC: ',[k+':'+str(v/e_tm) for k,v in time_spent.items()])
        import numpy as np 
        if verbose: print('rl = ',np.dot(np.array(ui),np.array(pi)))
        rl = np.dot(np.array(ui),np.array(pi))
        # print('reliability of T batches= ', rl)
        # for i,j,k in zip(states,ui,pi):
        #     print('stratum: ',i)
        #     print('mu_i: ', j)
        #     print('pi_i: ', k)

        return rl, ui, nbrs

    def bfs_sample(self,source,target, seed = 1, optimiseargs = {'nbrs':None, 'doopt': False}, verbose = False):
        """ For Reachability query. """
        # print(self.Edges)
        # print('bfs_sample: seed = ',seed)
        start_execution_time = time()
        
        assert len(self.nbrs)>0
        if optimiseargs is not None:
            if optimiseargs['nbrs'] is None:
                nbrs = self.construct_nbrs() # Constructs node => Nbr incidence dictionary. 
            else:
                nbrs = optimiseargs['nbrs']
        else:
            nbrs = self.construct_nbrs()
        if source not in nbrs or target not in nbrs:
            return nbrs, [], 0, 0
        
        queue = deque([source]) # Should be deque()
        reached_target = 0
        # if verbose:
        sample = [] # Remove this container
        prob_sample = 1.0
        random.seed(seed)
        visited = {source: True}
        # verbose = True
        while len(queue) and reached_target == 0: # MC+BFS loop
            # u = queue.pop(0)
            u = queue.popleft()
            # visited[u] = True
            if u == target:
                reached_target = 1
                break
            for v in nbrs.get(u,[]):
                (uu,vv) = (min(u,v),max(u,v))
                p = self.edict.get((uu,vv),-1)
                if p == -1: #
                    # print(sample,'\n',(uu,vv),' ',u) 
                    continue 
                if random.random() < p:
                    if (not visited.get(v,False)):
                        visited[v] = True
                        if verbose:
                            sample.append((uu,vv))
                            prob_sample *= p 
                        queue.append(v)
                        if v == target:
                            reached_target = 1
                            break 
                else:
                    if verbose: prob_sample *= (1-p)
        support_value = reached_target
        sample_tm = time() - start_execution_time
        self.total_sample_tm += sample_tm
        # print(sorted(sample))
        # print(source,target,sample,support_value)
        return nbrs, sample, prob_sample,support_value 
        # return sample, prob_sample,support_value # possible world G, Pr(G), Reach/Not
    
    def dijkstra_sample(self,source,target, seed = 1, optimiseargs = {'nbrs':None, 'doopt': False}, verbose = False):
        """ For SP query (unweighted graph). """
        # print(self.Edges)
        # print('simple graph dijkstra')
        start_execution_time = time()
        assert len(self.nbrs) > 0
        if optimiseargs is not None:
            if optimiseargs['nbrs'] is None:
                nbrs = self.construct_nbrs() # Constructs node => Nbr incidence dictionary. 
            else:
                nbrs = optimiseargs['nbrs']
        else:
            nbrs = self.construct_nbrs()
        if source not in nbrs or target not in nbrs:
            return nbrs, [], 0, INFINITY
        
        reached_target = 0
        # if verbose:
        sample = [] # No need this.
        prob_sample = 1.0 # No need.
        random.seed(seed)
        seen = {source:0}
        dists = {}
        heap = []
        heappush(heap,(0,source))
        while len(heap) and reached_target == 0: # MC+BFS loop
            dist_u, u = heappop(heap)
            if u in dists:
                continue 
            dists[u] = dist_u
            if u == target:
                reached_target = 1
                break
            for v in nbrs.get(u,[]):
                (uu,vv) = (min(u,v),max(u,v))
                dist_uv = dists[u] + self.weights[(uu,vv)]
                p = self.edict.get((uu,vv),-1)
                if p == -1: # unexpected edge.
                    # print(sample,'\n',(uu,vv),' ',u) 
                    continue 
                if random.random() < p:
                    if (v not in seen) or (dist_uv < seen[v]):
                        seen[v] = dist_uv
                        if verbose:
                            sample.append((uu,vv))
                            prob_sample *= p 
                        heappush(heap,(dist_uv,v))
                        if v == target:
                            reached_target = 1
                            dists[v] = dist_uv
                            break 
                else:
                    if verbose:     prob_sample *= (1-p)
        support_value = dists.get(target,INFINITY)
        # print(source,target,sample,support_value,dists)
        # return sample, prob_sample,support_value # possible world G, Pr(G), Reach/Not
        sample_tm = time() - start_execution_time
        self.total_sample_tm += sample_tm
        return nbrs, sample, prob_sample,support_value 

    def dhop_reach_sample(self,source,target, seed = 1, dhop = None, optimiseargs = {'nbrs':None, 'doopt': False}, verbose = False):
        """ For SP query (unweighted graph). """
        # print(self.Edges)
        start_execution_time = time()
        if len(self.nbrs) == 0:
            self.nbrs = self.construct_nbrs()
        if optimiseargs is not None:
            if optimiseargs['nbrs'] is None:
                nbrs = self.construct_nbrs() # Constructs node => Nbr incidence dictionary. 
            else:
                nbrs = optimiseargs['nbrs']
        else:
            nbrs = self.construct_nbrs()
        
        if dhop is None:
            max_d = INFINITY
        else:
            max_d = dhop 
            
        reached_target = 0
        sample = []
        prob_sample = 1.0
        random.seed(seed)
        # print('seed: ',seed)
        seen = {source:0}
        dists = {}
        heap = []
        heappush(heap,(0,source))
        while len(heap) and reached_target == 0: # MC+BFS loop
            dist_u, u = heappop(heap)
            if u in dists:
                continue 
            dists[u] = dist_u
            if u == target and dist_u <= max_d:
                reached_target = 1
                break
            for v in nbrs.get(u,[]):
                (uu,vv) = (min(u,v),max(u,v))
                dist_uv = dists[u] + 1
                p = self.edict.get((uu,vv),-1)
                if p == -1: # unexpected edge.
                    # print(sample,'\n',(uu,vv),' ',u) 
                    continue 
                if random.random() < p:
                    if (v not in seen) or (dist_uv < seen.get(v,INFINITY)):
                        seen[v] = dist_uv
                        if verbose:
                            sample.append((uu,vv))
                            prob_sample *= p 
                        heappush(heap,(dist_uv,v))
                        if v == target and dist_uv <= max_d:
                            reached_target = 1
                            # dists[v] = dist_uv
                            break 
                else:
                    if verbose:     prob_sample *= (1-p)
        support_value = reached_target
        # print(source,target,sample,support_value,dists)
        # return sample, prob_sample,support_value # possible world G, Pr(G), Reach/Not
        sample_tm = time() - start_execution_time
        self.total_sample_tm += sample_tm
        # print(sample)
        # print(support_value)
        return nbrs, sample, prob_sample,support_value 
    
    def add_edge(self,u,v,prob,weight = 1.0, construct_nbr = False, construct_nx = False):
        """ Add edge e = (u,v) along with p(e) """
        (u,v) = (min(u,v),max(u,v))
        if construct_nx:
            self.nx_format.add_edge(u,v)
        else:
            self.Edges.append((u,v))
            self.edict[(u,v)] = prob 
            # self.notedict[(u,v)] = 1-prob # [ Not necessary ]
            if weight is not None:
                self.weights[(u,v)] = weight
            else:
                self.weights[(u,v)] = 1
        if construct_nbr:
            _tmp = self.nbrs.get(u,[])
            _tmp.append(v)
            self.nbrs[u] = _tmp 
            _tmp = self.nbrs.get(v,[])
            _tmp.append(u)
            self.nbrs[v] = _tmp
        

    def get_next_prob(self,u,v,type = 'o1'):
        ''' returns p_{up}(e): the probability of (u,v) after update without actually materialising it.  '''
        if type == 'o1':
            return 1
        elif type == 'o2':
            prob_prev = self.edict[(u,v)]
            if prob_prev>=0.5:
                return 1
            else:
                return 0
        elif type == 'u2C':
            return [0,1]

    def edge_update(self, u, v, type = 'o1'):
        '''  Updates the probability of an edge Given an op type '''
        (u,v) = (min(u,v),max(u,v))
        # assert ((u,v) in self.edict and (u,v) in self.notedict)
        assert ((u,v) in self.edict)
        self.update_edge_prob(u,v, self.get_next_prob(u,v,type))

    def update_edge_prob(self, u, v, prob):
        """ 
        Updates the probability of an edge to a given value
        """
        # if type(prob) is not list:
        (u,v) = (min(u,v),max(u,v))
        # assert ((u,v) in self.edict and (u,v) in self.notedict)
        assert ((u,v) in self.edict)
        self.edict[(u,v)] = prob  
        # self.notedict[(u,v)] = 1 - prob
        # else: # stochastic update
        #     self.edict0 = deepcopy(self.edict)
        #     self.edict0[(u,v,id)]  = 0 # Cleaned to 0
        #     self.edict1 = deepcopy(self.edict)
        #     self.edict1[(u,v,id)] = 1 # Cleaned to 1
        #     return ([self.edict0,self.edict1])
      
    def enumerate_worlds(self):
        """
        Explicitely enumerates all possible worlds.
        Returns G, Pr(G)
        """
        # start_execution_time = time()
        m = len(self.Edges)
        bit_vec = range(m)
        for _len in range(0, m+1):
            for sub_bitvec in itertools.combinations(bit_vec,_len):
                sub_bitvec_s = set(sub_bitvec)
                poss_world = []
                poss_world_prob = 1
                # start_execution_time = time()
                for i,e in enumerate(self.Edges):
                    if i in sub_bitvec_s:
                        poss_world.append(e)
                        poss_world_prob *= self.edict[e]
                    else:
                        poss_world_prob *= (1-self.edict[e])
                # [(e,[self.notedict[e],self.edict[e]][i in sub_bitvec]) for i,e in enumerate(self.Edges)] 
                # self.sample_time_list.append(time()-start_execution_time)
                yield (poss_world, poss_world_prob)
        # self.sample_time_list.append(time()-start_execution_time)
    
    # def enumerate_worlds(self):
    #     """
    #     O(N) space solution using backtracking
    #     Explicitely enumerates all possible worlds.
    #     Returns G, Pr(G)
    #     """
    #     # print('enumerate worlds: ')
    #     m = len(self.Edges)
    #     def backtrack(k, first = 0, curr = []):
    #         # if the combination is done
    #         if len(curr) == k:  
    #             yield curr[:]
            
    #         for i in range(first, m):
    #             # add nums[i] into the current combination
    #             curr.append(i)
    #             # use next integers to complete the combination
    #             yield from backtrack(k,i + 1, curr)
    #             # backtrack
    #             curr.pop()
    #     master_set = set(range(0,m))
    #     for k in range(0, m+1):
    #         # print(k)
    #         for G in backtrack(k):
    #             # print('G = ',G)
    #             set_G = set(G)
    #             Gprime = master_set - set_G
    #             poss_world = []
    #             poss_world_prob = 1
    #             # for i,e in enumerate(self.Edges):
    #             #     if i in G:
    #             #         poss_world.append(e)
    #             #         poss_world_prob = poss_world_prob * self.edict[e]
    #             #     else:
    #             #         poss_world_prob = poss_world_prob * self.notedict[e]
    #             for i in set_G:
    #                 # poss_world.append(self.Edges[i])
    #                 poss_world_prob *= self.edict[self.Edges[i]]
    #             for i in Gprime:
    #                 poss_world_prob *= self.notedict[self.Edges[i]]
    #             yield (poss_world, poss_world_prob)
    #     # self.sample_time_list.append(time()-start_execution_time)
    
    # def enumerate_worlds2(self):
    #     """
    #     Explicitely enumerates all possible worlds and returns G, Pr(G), sum p(e) for all e \in G
    #     """
    #     m = len(self.Edges)
    #     bit_vec = range(m)
    #     for _len in range(0, m+1):
    #         for sub_bitvec in itertools.combinations(bit_vec,_len):
    #             poss_world = []
    #             poss_world_prob = 1
    #             _sum = 0
    #             for i,e in enumerate(self.Edges):
    #                 if i in sub_bitvec:
    #                     poss_world.append(e)
    #                     poss_world_prob = poss_world_prob * self.edict[e]
    #                     _sum += self.edict[e]
    #                 else:
    #                     poss_world_prob = poss_world_prob * self.notedict[e]
    #             yield (poss_world, poss_world_prob,_sum)

    def enumerate_k_edges(self, k = 1):
        """ Generate all possible k-subsets of the Edges. """
        m = len(self.Edges)
        bit_vec = range(m)
        for sub_bitvec in itertools.combinations(bit_vec, k):
            subset = []
            for i,e in enumerate(self.Edges):
                if i in sub_bitvec:
                    subset.append(e)
            yield subset

    def plot_probabilistic_graph(self, seed = 1, savedrawing = False, filename = 'ug_undirected.pdf', title = 'Uncertain graph'):
        """
        Plots the uncertain graph as a weighted graph.
        """
        fig,ax = plt.subplots()
        
        nx_graph = nx.Graph()
        for edge in self.Edges:
            nx_graph.add_edge(*edge,weight = self.edict[edge])
#         print(nx_graph.nodes)
#         print(nx_graph.edges)
        pos = nx.spring_layout(nx_graph, seed = seed, weight = None)

        nx.draw_networkx_nodes(nx_graph, pos = pos, ax = ax)

        nx.draw_networkx_edges(nx_graph, pos = pos, edgelist = nx_graph.edges, ax = ax)

        labels = nx.get_edge_attributes(nx_graph,'weight')
        nx.draw_networkx_labels(nx_graph, pos = pos, ax = ax, font_size = 14)
        nx.draw_networkx_edge_labels(nx_graph,pos, ax = ax, edge_labels=labels, font_size = 14)
        plt.title(title)
        fig.tight_layout()
        if savedrawing:
            plt.savefig(filename, bbox_inches = 'tight')
        plt.show()
    
    def plot_possible_world_distr(self, saveplot = False):
        """ Plots the Probabilities (Y-axis) of individual possible worlds (X-axis) """
        pencolor = 'k' #black/red/blue etc
        penstyle = 'solid' # strock-style: solid/dashed etc.
        _temp = []
        for i,(world, prob) in enumerate(self.enumerate_worlds()):
            _temp.append((prob,i))
#         _temp = sorted(_temp,reverse = True)
        X = []
        Y = []
        for y,x in _temp:
            X.append(x)
            Y.append(y)
        plt.bar(X,Y, label = 'freq. distr.', color = pencolor, ls = penstyle)
        plt.ylim((0,1))
        plt.xlabel('Possible Worlds')
        plt.ylabel('Prob.')
        plt.legend()
        if saveplot:
             plt.savefig('words.pdf', bbox_inches = 'tight')
        plt.show()
    
    def get_num_edges(self):
        return len(self.Edges)
    
    def get_num_vertices(self):
        V = set()
        for e in self.Edges:
            V.add(e[0])
            V.add(e[1])
        return len(V)

    def reachable(self, source, target):
        queue = deque([source])
        reached_target = 0
        visited = {source: True}
        while len(queue) and reached_target == 0: 
            u = queue.popleft()
            if u == target:
                reached_target = 1
                break
            for v in self.nbrs.get(u,[]):
                if (not visited.get(v,False)):
                    visited[v] = True
                    queue.append(v)
                    if v == target:
                        reached_target = 1
                        break 
        return reached_target
    def dhop_reachable(self, source, target, d):
        reached_target = 0
        seen = {source:0}
        dists = {}
        heap = []
        heappush(heap,(0,source))
        while len(heap) and reached_target == 0: 
            dist_u, u = heappop(heap)
            if u in dists:
                continue 
            dists[u] = dist_u
            if u == target and dist_u <= d:
                reached_target = 1
                break
            for v in self.nbrs.get(u,[]):
                (uu,vv) = (min(u,v),max(u,v))
                dist_uv = dists[u] + 1
                if (v not in seen) or (dist_uv < seen[v]):
                    seen[v] = dist_uv
                    heappush(heap,(dist_uv,v))
                    if v == target and dist_uv <= d:
                        reached_target = 1
                        dists[v] = dist_uv
                        break 
        return reached_target
    
    def get_sample(self, seed = 1, verbose = False):
        """ Returns a random possible world (as a networkx Graph) instance. """
        start_execution_time = time()
        random.seed(seed)
        # nx_graph = nx.Graph()
        # poss_world = []
        poss_world = UGraph()
        poss_world_prob = 1
        for e in self.Edges:
            p = self.edict[e]
            # print(e,p)
            if random.random() < p:
                # nx_graph.add_edge(*e,weight = p)
                # poss_world.append(e)
                # poss_world.add_edge(e[0],e[1],p,self.weights[e],construct_nx=True)
                if len(self.weights) == 0:
                    poss_world.add_edge(e[0],e[1],p,weight=None,construct_nbr=True)
                else:
                    poss_world.add_edge(e[0],e[1],p,self.weights[e],construct_nbr=True)
                if verbose:
                    poss_world_prob = poss_world_prob * p
            else:
                if verbose:
                    poss_world_prob = poss_world_prob * (1-p)
        if verbose:
            # print(nx_graph.nodes)
            # print(nx_graph.edges)
            print(poss_world.Edges)
            print(poss_world_prob)
        sample_tm = time() - start_execution_time
        # self.sample_time_list.append(sample_tm)
        self.total_sample_tm += sample_tm
        return (poss_world,poss_world_prob) 

    def get_Ksample(self, K, seed = None):
        """ Returns a sample of K possible worlds """
        for i in range(K):
            if seed:
                yield self.get_sample(seed = seed+i,verbose = False)
            else:
                yield self.get_sample(seed = i,verbose = False)

    def get_Ksample_bfs(self, K=1, seed = None,source = None, target = None,\
                        optimiseargs = {'nbrs':None, 'doopt': False}):
        """ Returns a sample of K possible worlds """
        assert (source is not None and target is not None)
        for i in range(K):
            if seed:
                yield self.bfs_sample(seed = seed+i,source = source, target = target, optimiseargs=optimiseargs)
            else:
                yield self.bfs_sample(seed = i,source = source, target = target,optimiseargs=optimiseargs)
    
    def get_Ksample_dhopbfs(self, K=1, seed = None,source = None, target = None,\
                        dhop = None, optimiseargs = {'nbrs':None, 'doopt': False}):
        """ Returns a sample of K possible worlds """
        assert (source is not None and target is not None)
        for i in range(K):
            # print("seed ", seed)
            if seed:
                yield self.dhop_reach_sample(source, target, seed = seed+i, dhop = dhop, \
                                      optimiseargs=optimiseargs)
            else:
                yield self.dhop_reach_sample(source, target, seed = i, dhop = dhop,\
                                      optimiseargs=optimiseargs)
                
    def get_Ksample_dij(self, K=1, seed = None,source = None, target = None,\
                        optimiseargs = {'nbrs':None, 'doopt': False}):
        """ Returns a sample of K possible worlds """
        assert (source is not None and target is not None)
        for i in range(K):
            if seed:
                yield self.dijkstra_sample(seed = seed+i,source = source, target = target,\
                                           optimiseargs=optimiseargs)
            else:
                yield self.dijkstra_sample(seed = i,source = source, target = target,\
                                           optimiseargs=optimiseargs)

    def get_unweighted_graph_rep(self):
        """ 
        Returns the unweighted graph representation of the uncertain graph. 
        (Needed for Generating example queries in Generate_queries.py) 
        """
        nx_graph = nx.Graph()
        for edge in self.Edges:
            e = (edge[0],edge[1])
            nx_graph.add_edge(*e)
        return nx_graph 

    def get_unweighted_simple_graph_rep(self):
        """ 
        Returns the unweighted graph representation of the uncertain graph. 
        (Needed for Generating example queries in Generate_queries.py) 
        """
        nx_graph = nx.Graph()
        for edge in self.edict:
            e = (edge[0],edge[1])
            nx_graph.add_edge(*e)
        return nx_graph 
    def get_weighted_graph_rep(self):
        """ 
        Returns the unweighted graph representation of the uncertain graph. 
        (Needed for Generating example queries in Generate_queries.py) 
        """
        nx_graph = nx.Graph()
        for edge,prob in self.edict.items():
            e = (edge[0],edge[1])
            nx_graph.add_edge(*e,weight = self.weights[edge], prob = prob)
        return nx_graph 

class UMultiGraph(UGraph):
    """
    A generic class for Uncertain Graph data structure and for various operations on it. 
    """
    def __init__(self):
        super().__init__()
        self.nx_format = nx.MultiGraph()
    
    def simplify(self, construct_nx = False):
        """ 
        Makes the graph simple by removing repeated edges while keeping
        only the edge with smallest weight.
        """
        sg = UGraph()
        _small = {}
        for u,v,_id in self.Edges:
            if (u,v) in _small:
                if _small[(u,v)] > self.weights[(u,v,_id)]:
                    w_uv = self.weights[(u,v,_id)]
                    if (u,v) in sg.edict:
                        sg.update_edge_prob(u,v,self.edict[(u,v,_id)])
                        sg.weights[(u,v)] = w_uv
                    else:
                        if construct_nx:
                            sg.add_edge(u,v,self.edict[(u,v,_id)],w_uv,construct_nx=True)
                        else:
                            sg.add_edge(u,v,self.edict[(u,v,_id)],w_uv,construct_nbr=True)
                    _small[(u,v)] = w_uv
            else:
                w_uv = self.weights[(u,v,_id)]
                _small[(u,v)] = w_uv
                if construct_nx:
                    sg.add_edge(u,v,self.edict[(u,v,_id)],w_uv,construct_nx=True)
                else:
                    sg.add_edge(u,v,self.edict[(u,v,_id)],w_uv,construct_nbr=True)
        if construct_nx:
            return sg.nx_format
        else:
            return sg

    def get_prob(self, e):
        u,v = min(e[0],e[1]),max(e[0],e[1]) 
        return self.edict[(u,v,e[2])]
    
    def __str__(self):
        return str(self.Edges)
    
    def add_edge(self,u,v,id, prob, weight = 1.0, construct_nbr = False, construct_nx = False):
        """ Add edge e = (u,v) along with p(e) """
        (u,v) = (min(u,v),max(u,v))
        if construct_nx:
            self.nx_format.add_edge(u,v)
        else:
            self.Edges.append((u,v,id))
            self.edict[(u,v,id)] = prob 
            # self.notedict[(u,v,id)] = 1-prob 
            self.weights[(u,v,id)] = weight
        if construct_nbr: # here neighbors are not duplicated
            _tmp = self.nbrs.get(u,[])
            _tmp.append((v,id))
            self.nbrs[u] = _tmp 
            _tmp = self.nbrs.get(v,[])
            _tmp.append((u,id))
            self.nbrs[v] = _tmp

    def get_next_prob(self,u,v,id,type = 'o1'):
        ''' returns p_{up}(e): the probability of (u,v) after update without actually materialising it. '''
        if type == 'o1':
            return 1
        elif type == 'o2':
            prob_prev = self.edict[(u,v,id)]
            if prob_prev>=0.5:
                return 1
            else:
                return 0
        elif type == 'u2C':
            return [0,1]

    def edge_update(self, u, v, id, type = 'o1'):
        '''  Updates the probability of an edge Given an op type '''
        (u,v) = (min(u,v),max(u,v))
        # assert ((u,v,id) in self.edict and (u,v,id) in self.notedict)
        assert ((u,v,id) in self.edict)
        self.update_edge_prob(u, v, id, self.get_next_prob(u,v,id,type))

    def update_edge_prob(self, u, v, id, prob):
        """ 
        Updates the probability of an edge to a given value
        """
        if type(prob) is not list: # Deterministic update
            (u,v) = (min(u,v),max(u,v))
            # assert ((u,v,id) in self.edict and (u,v,id) in self.notedict)
            assert ((u,v,id) in self.edict )
            self.edict[(u,v,id)] = prob  
            # self.notedict[(u,v,id)] = 1 - prob
        else: # stochastic update
            self.edict0 = deepcopy(self.edict)
            self.edict0[(u,v,id)]  = 0 # Cleaned to 0
            self.edict1 = deepcopy(self.edict)
            self.edict1[(u,v,id)] = 1 # Cleaned to 1
            return ([self.edict0,self.edict1])
    def plot_probabilistic_graph(self, seed = 1, savedrawing = False, filename = 'ug_undirected.pdf', title = 'Uncertain graph'):
        pass 

    def plot_possible_world_distr(self, saveplot = False):
        pass
    
    def construct_nbrs(self):
        if len(self.nbrs) == 0:
            nbrs = {} # key = vertex id, value = [list of edge ids]
            for e in self.Edges:
                u,v,_id = e
                _tmp = nbrs.get(u,[])
                _tmp.append((v,_id))
                nbrs[u] = _tmp 
                _tmp = nbrs.get(v,[])
                _tmp.append((u,_id))
                nbrs[v] = _tmp 
            self.nbrs = nbrs
        else:
            nbrs = self.nbrs
        return nbrs 
    
    def get_sample(self, seed = 1, verbose = False, simplify = True):
        """ Returns a random possible world instance. """
        # print('MC sample multigraph')
        start_execution_time = time()
        random.seed(seed)
        if simplify:
            poss_world = nx.Graph()
        else:
            poss_world = UMultiGraph()
        poss_world_prob = 1
        for e in self.Edges:
            p = self.edict[e]
            # print(e,p)
            if random.random() < p:
                if simplify:
                    u,v = min(e[0],e[1]),max(e[0],e[1])
                    if poss_world.has_edge(u,v): # parallel edge to an already added edge
                        if poss_world[u][v]['weight'] > self.weights[e]:
                            poss_world[u][v]['weight'] = self.weights[e]
                    else:
                        poss_world.add_edge(u, v, weight=self.weights[e])
                else:
                    poss_world.add_edge(e[0],e[1],p,self.weights[e])
        sample_tm = time() - start_execution_time
        self.total_sample_tm += sample_tm
        return (poss_world,poss_world_prob) 

    # def bfs_sample(self,source,target, seed = 1,optimiseargs = {'nbrs':None, 'doopt': False}, verbose = False):
    #     """ For Reachability query. """
    #     print('bfs_sample multigraph')
    #     # print(self.Edges)
    #     start_execution_time = time()
    #     assert len(self.nbrs)>0
    #     if optimiseargs is not None:
    #         if optimiseargs['nbrs'] is None:
    #             nbrs = self.construct_nbrs() # Constructs node => Nbr incidence dictionary. 
    #         else:
    #             nbrs = optimiseargs['nbrs']
    #     else:
    #         nbrs = self.construct_nbrs()
    #     # print(nbrs)
    #     queue = deque([source])
    #     reached_target = 0
    #     sample = []
    #     prob_sample = 1.0
    #     random.seed(seed)
    #     visited = {}
    #     while len(queue) and reached_target == 0: # MC+BFS loop
    #         # u = queue.pop(0)
    #         u = queue.popleft()
    #         # print('pop: ',u)
    #         # visited[u] = True
    #         if u == target:
    #             reached_target = 1
    #             break
    #         for v, eid in nbrs.get(u,[]):
    #             # print('traversing ',v,' via ',eid)
    #             (uu,vv) = (min(u,v),max(u,v))
    #             p = self.edict.get((uu,vv,eid),-1)
    #             if p == -1: #
    #                 # print(sample,'\n',(uu,vv),' ',u) 
    #                 continue 
    #             if random.random() < p:
    #                 if (not visited.get(eid,False)):
    #                     visited[eid] = True
    #                     if verbose: 
    #                         sample.append((uu,vv))
    #                         # print('added ',(uu,vv),' into poss world')
    #                         prob_sample *= p 
    #                     queue.append(v)
    #                     if v == target:
    #                         reached_target = 1
    #                         break 
    #             else:
    #                 if verbose:     prob_sample *= (1-p)
    #             # print(visited)
    #     support_value = reached_target
    #     sample_tm = time() - start_execution_time
    #     self.total_sample_tm += sample_tm
    #     # print(source,target,sample,support_value)
    #     return nbrs, sample, prob_sample,support_value # possible world G, Pr(G), Reach/Not

    def dijkstra_sample(self,source,target, seed = 1,optimiseargs = {'nbrs':None, 'doopt': False}, verbose = False):
        """ For SP query (unweighted graph). """
        # print(self.Edges)
        # print('dijkstra sample multigraph')
        start_execution_time = time()
        assert len(self.nbrs)>0
        if optimiseargs is not None:
            if optimiseargs['nbrs'] is None:
                nbrs = self.construct_nbrs() # Constructs node => Nbr incidence dictionary. 
            else:
                nbrs = optimiseargs['nbrs']
        else:
            nbrs = self.construct_nbrs()
        if source not in nbrs or target not in nbrs:
            return nbrs, [], 0, INFINITY
        reached_target = 0
        sample = []
        prob_sample = 1.0
        random.seed(seed)
        seen = {source:0}
        # seen = {}
        dists = {}
        heap = []
        heappush(heap,(0,source))
        while len(heap) and reached_target == 0: # MC+BFS loop
            dist_u, u = heappop(heap)
            if u in dists:
                continue 
            dists[u] = dist_u
            if u == target:
                reached_target = 1
                break
            for v,eid in nbrs.get(u,[]):
                (uu,vv) = (min(u,v),max(u,v))
                p = self.edict.get((uu,vv,eid),-1)
                if p == -1: # unexpected edge.
                    # print(sample,'\n',(uu,vv),' ',u) 
                    continue 
                if random.random() < p:
                    dist_uv = dists[u] + self.weights[(uu,vv,eid)]
                    if (v not in seen) or (dist_uv < seen[v]):
                        seen[v] = dist_uv
                        if verbose:
                            sample.append((uu,vv))
                            prob_sample *= p 
                        heappush(heap,(dist_uv,v))
                        if v == target:
                            reached_target = 1
                            dists[v] = dist_uv
                            break 
                else:
                    if verbose: prob_sample *= (1-p)
        support_value = dists.get(target,INFINITY)
        sample_tm = time() - start_execution_time
        self.total_sample_tm += sample_tm
        # print(source,target,sample,support_value,dists)
        return nbrs, sample, prob_sample,support_value # possible world G, Pr(G), Reach/Not
    
    # def dhop_reach_sample(self,source,target, seed = 1, dhop = None, optimiseargs = {'nbrs':None, 'doopt': False}, verbose = False):
    #     """ For d-hop reach query (multi graph). """
    #     # print(self.Edges)
    #     print('d-hop reachability sample multigraph')
    #     assert len(self.nbrs)>0
    #     start_execution_time = time()
    #     if optimiseargs is not None:
    #         if optimiseargs['nbrs'] is None:
    #             nbrs = self.construct_nbrs() # Constructs node => Nbr incidence dictionary. 
    #         else:
    #             nbrs = optimiseargs['nbrs']
    #     else:
    #         nbrs = self.construct_nbrs()
        
    #     if dhop is None:
    #         max_d = INFINITY
    #     else:
    #         max_d = dhop 
            
    #     reached_target = 0
    #     sample = []
    #     prob_sample = 1.0
    #     # print('seed: ',seed)
    #     random.seed(seed)
    #     seen = {source:0}
    #     dists = {}
    #     heap = []
    #     heappush(heap,(0,source))
    #     while len(heap) and reached_target == 0: # MC+BFS loop
    #         dist_u, u = heappop(heap)
    #         if u in dists:
    #             continue 
    #         dists[u] = dist_u
    #         if u == target and dist_u <= max_d:
    #             reached_target = 1
    #             break
    #         for v,eid in nbrs.get(u,[]):
    #             (uu,vv) = (min(u,v),max(u,v))
    #             dist_uv = dists[u] + self.weights[(uu,vv,eid)]
    #             p = self.edict.get((uu,vv,eid),-1)
    #             if p == -1: # unexpected edge.
    #                 # print(sample,'\n',(uu,vv),' ',u) 
    #                 continue 
    #             if random.random() < p:
    #                 if (v not in seen) or (dist_uv < seen[v]):
    #                     seen[v] = dist_uv
    #                     if verbose:
    #                         sample.append((uu,vv))
    #                         prob_sample *= p 
    #                     heappush(heap,(dist_uv,v))
    #                     if v == target and dist_uv <= max_d:
    #                         reached_target = 1
    #                         dists[v] = dist_uv
    #                         break 
    #             else:
    #                 if verbose: prob_sample *= (1-p)
    #     support_value = reached_target
    #     # print(source,target,sample,support_value,dists)
    #     # return sample, prob_sample,support_value # possible world G, Pr(G), Reach/Not
    #     sample_tm = time() - start_execution_time
    #     self.total_sample_tm += sample_tm
    #     return nbrs, sample, prob_sample,support_value 
    
    # def find_rel_rss(self,T, source,target,seed = 1, optimiseargs = {'nbrs':None, 'doopt': False}):
    #     # print('source: ',source, ' target: ',target)
    #     print('rss sample multigraph')
    #     kRecursiveSamplingThreshold = 5
    #     states = ["0001","0010","0011","0100","0101",\
    #               "0110","0111","1000","1001","1010",\
    #              "1011","1100","1101","1110","1111",\
    #             "0000"]
    #     start_execution_time = time()
    #     assert len(self.nbrs)>0
    #     if optimiseargs is not None:
    #         if optimiseargs['nbrs'] is None:
    #             nbrs = self.construct_nbrs() # Constructs node => Nbr incidence dictionary. 
    #         else:
    #             nbrs = optimiseargs['nbrs']
    #     else:
    #         nbrs = self.construct_nbrs()
    #     if (source not in nbrs) or (target not in nbrs):
    #         return 0.0,nbrs
    #     # print(nbrs)
    #     sv_map = set([source])
    #     si_queue = [(source,t,eid) for t,eid in nbrs[source]]
    #     # print('si_q: ',si_queue, ': ',nbrs[source])
    #     random.seed(seed)
    #     def samplingR_RSS(sv_map, si_queue):
    #         temp_q = []
    #         while len(si_queue):
    #             e = si_queue.pop(0)
    #             p = self.get_prob(e)
    #             if e[1] in sv_map:
    #                 continue 
    #             if random.random() < p:
    #                 if e[1] == target:
    #                     return 1 
    #                 sv_map.add(e[1])
    #                 temp_q.append(e[1])
    #         while len(temp_q):
    #             v = temp_q.pop(0)
    #             nbrs_v = nbrs[v]
    #             if len(nbrs_v):
    #                 for w,eid in nbrs_v:
    #                     p_vw = self.get_prob((v,w,eid))
    #                     if random.random() < p_vw:
    #                         if w == target:
    #                             return 1 
    #                         if w not in sv_map:
    #                             sv_map.add(w)
    #                             temp_q.append(w)
    #         return 0 
        
    #     def findReliability_RHH_forRSS(sv_map, si_queue, n, flag, node):
    #         # print('->', sv_map,si_queue, n, flag, node)
    #         if flag:
    #             if node == target:
    #                 return 1 
    #             sv_map.add(node)
    #             si_queue += [(node,t,eid) for t,eid in nbrs[node]]
    #         if len(si_queue)==0:
    #             return 0
    #         if len(si_queue)>4:
    #             findReliability_RSS(sv_map,si_queue,n,15,[])
    #         if n <= kRecursiveSamplingThreshold:
    #             if n == 0:
    #                 return 0 
    #             rhh = 0
    #             for i in range(n):
    #                 r = samplingR_RSS(deepcopy(sv_map),deepcopy(si_queue))
    #                 # print(i,' : ',r)
    #                 rhh += r
    #             return rhh/n
    #         e = si_queue.pop(0)
    #         while (e[1] in sv_map):
    #             if len(si_queue) == 0:
    #                 return 0 
    #             e = si_queue.pop(0)
    #         # print('->',e)
    #         pe = self.get_prob(e)
    #         # print(' ->-> ', math.floor(n*pe), n- math.floor(n*pe))
    #         return pe * findReliability_RHH_forRSS(deepcopy(sv_map),deepcopy(si_queue),floor(n*pe),True,e[1]) + \
    #         (1- pe)* findReliability_RHH_forRSS(deepcopy(sv_map),deepcopy(si_queue), n - floor(n*pe),False,e[1]) 
        
    #     def findReliability_RSS(sv_map, si_queue, n, flag, nodes):
    #         for i in range(4):
    #             if (states[flag][i] == '1'):
    #                 if (nodes[i] == target):
    #                     return 1           
    #                 sv_map.add(nodes[i])
    #                 si_queue += [(nodes[i],t,eid) for t,eid in nbrs[nodes[i]]]
    #         if len(si_queue)==0:
    #             return 0
            
    #         if n <= kRecursiveSamplingThreshold:
    #             if (n <= 0):
    #                 return 0
    #             rhh = 0
    #             for i in range(n): #Run n times
    #                 rhh += samplingR_RSS(deepcopy(sv_map), deepcopy(si_queue))
    #             return rhh*1.0 / n
    #         # print('si_quue: ',si_queue)
    #         nodes = []
    #         edges= []
    #         while len(nodes) < 4:
    #             if len(si_queue) == 0:
    #                 si_queue += [e for e in edges]
    #                 # print(si_queue, sv_map, nodes,target)
    #                 return findReliability_RHH_forRSS(deepcopy(sv_map), deepcopy(si_queue), n, False, target)
    #             e = si_queue.pop(0)
    #             if e[1] in sv_map:
    #                 continue 
    #             else:
    #                 nodes.append(e[1])
    #                 edges.append(e)

    #         reliablity= 0
    #         temp_n = n 
    #         temp_prob = 1.0 
    #         probs = []
    #         for i in range(4):
    #             probs.append(self.get_prob(edges[i]))
    #         for i in range(16):
    #             if i == 15:
    #                 reliablity += temp_prob * findReliability_RSS(deepcopy(sv_map), deepcopy(si_queue), temp_n, 15, nodes)
    #                 break 
    #             prob = 1.0
    #             for j in range(4):
    #                 if states[i][j] == '1':
    #                     prob = prob * probs[j]
    #                 else:
    #                     prob = prob * (1-probs[j])
    #             this_n = floor(n*prob)
    #             temp_n -= this_n 
    #             temp_prob -= prob 
    #             reliablity += (prob * findReliability_RSS(deepcopy(sv_map), deepcopy(si_queue), this_n, i, nodes))
    #         return reliablity
    
    #     return findReliability_RSS(sv_map,si_queue, n = T,flag = 15, nodes = []), nbrs
