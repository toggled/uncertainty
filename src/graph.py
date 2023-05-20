import itertools 
import networkx as nx
from matplotlib import pyplot as plt 
import random 
from time import time 
from copy import deepcopy
from heapq import heappush, heappop
from networkx.algorithms.bipartite.matching import INFINITY
class UGraph:
    """
    A generic class for Uncertain Graph data structure and for various operations on it. 
    Supports edge weights
    """
    def __init__(self):
        self.Edges = [] # Edge list of the uncertain graph
        self.edict = {} # Edge -> Prob mapping of the uncertain graph.
        self.notedict = {} # Edge -> (1-Prob) mapping of the uncertain graph. Mostly to assist faster computation.
        # self.sample_time_list = [] # exec. time to generate individual possible worlds
        self.total_sample_tm = 0
        self.weights = {}
    def construct_nbrs(self):
        nbrs = {}
        for e in self.Edges:
            _tmp = nbrs.get(e[0],[])
            _tmp.append(e[1])
            nbrs[e[0]] = _tmp 
            _tmp = nbrs.get(e[1],[])
            _tmp.append(e[0])
            nbrs[e[1]] = _tmp 
        return nbrs 
    
    def bfs_sample(self,source,target, seed = 1, optimiseargs = {'nbrs':None, 'doopt': False}):
        """ For Reachability query. """
        # print(self.Edges)
        if optimiseargs is not None:
            if optimiseargs['nbrs'] is None:
                nbrs = self.construct_nbrs() # Constructs node => Nbr incidence dictionary. 
            else:
                nbrs = optimiseargs['nbrs']
        else:
            nbrs = self.construct_nbrs()
        queue = [source]
        reached_target = 0
        sample = []
        prob_sample = 1.0
        random.seed(seed)
        visited = {source: True}
        while len(queue) and reached_target == 0: # MC+BFS loop
            u = queue.pop(0)
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
                        sample.append((uu,vv))
                        prob_sample *= p 
                        queue.append(v)
                        if v == target:
                            reached_target = 1
                            break 
                else:
                    prob_sample *= (1-p)
        support_value = reached_target
        # print(source,target,sample,support_value)
        return nbrs, sample, prob_sample,support_value 
        # return sample, prob_sample,support_value # possible world G, Pr(G), Reach/Not
    
    def dijkstra_sample(self,source,target, seed = 1):
        """ For SP query (unweighted graph). """
        # print(self.Edges)
        nbrs = self.construct_nbrs() # Constructs node => [Nbr nodes list] incidence dictionary. 
        reached_target = 0
        sample = []
        prob_sample = 1.0
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
                        sample.append((uu,vv))
                        prob_sample *= p 
                        heappush(heap,(dist_uv,v))
                        if v == target:
                            reached_target = 1
                            dists[v] = dist_uv
                            break 
                else:
                    prob_sample *= (1-p)
        support_value = dists.get(target,INFINITY)
        # print(source,target,sample,support_value,dists)
        return sample, prob_sample,support_value # possible world G, Pr(G), Reach/Not

    def add_edge(self,u,v,prob,weight = 1.0):
        """ Add edge e = (u,v) along with p(e) """
        (u,v) = (min(u,v),max(u,v))
        self.Edges.append((u,v))
        self.edict[(u,v)] = prob 
        self.notedict[(u,v)] = 1-prob 
        self.weights[(u,v)] = weight

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
        assert ((u,v) in self.edict and (u,v) in self.notedict)
        self.update_edge_prob(u,v, self.get_next_prob(u,v,type))

    def update_edge_prob(self, u, v, prob):
        """ 
        Updates the probability of an edge to a given value
        """
        if type(prob) is not list:
            (u,v) = (min(u,v),max(u,v))
            assert ((u,v) in self.edict and (u,v) in self.notedict)
            self.edict[(u,v)] = prob  
            self.notedict[(u,v)] = 1 - prob
        else: # stochastic update
            self.edict0 = deepcopy(self.edict)
            self.edict0[(u,v,id)]  = 0 # Cleaned to 0
            self.edict1 = deepcopy(self.edict)
            self.edict1[(u,v,id)] = 1 # Cleaned to 1
            return ([self.edict0,self.edict1])
      
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
                        poss_world_prob *= self.notedict[e]
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
    
    def enumerate_worlds2(self):
        """
        Explicitely enumerates all possible worlds and returns G, Pr(G), sum p(e) for all e \in G
        """
        m = len(self.Edges)
        bit_vec = range(m)
        for _len in range(0, m+1):
            for sub_bitvec in itertools.combinations(bit_vec,_len):
                poss_world = []
                poss_world_prob = 1
                _sum = 0
                for i,e in enumerate(self.Edges):
                    if i in sub_bitvec:
                        poss_world.append(e)
                        poss_world_prob = poss_world_prob * self.edict[e]
                        _sum += self.edict[e]
                    else:
                        poss_world_prob = poss_world_prob * self.notedict[e]
                yield (poss_world, poss_world_prob,_sum)

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

    def get_sample(self, seed = 1, verbose = False):
        """ Returns a random possible world (as a networkx Graph) instance. """
        start_execution_time = time()
        random.seed(seed)
        # nx_graph = nx.Graph()
        poss_world = []
        poss_world_prob = 1
        for e in self.Edges:
            p = self.edict[e]
            # print(e,p)
            if random.random() < p:
                # nx_graph.add_edge(*e,weight = p)
                poss_world.append(e)
                poss_world_prob = poss_world_prob * p
            else:
                poss_world_prob = poss_world_prob * (1-p)
        if verbose:
            # print(nx_graph.nodes)
            # print(nx_graph.edges)
            print(poss_world)
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

    def get_Ksample_dij(self, K=1, seed = None,source = None, target = None):
        """ Returns a sample of K possible worlds """
        assert (source is not None and target is not None)
        for i in range(K):
            if seed:
                yield self.dijkstra_sample(seed = seed+i,source = source, target = target)
            else:
                yield self.dijkstra_sample(seed = i,source = source, target = target)

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

    def add_edge(self,u,v,id, prob, weight = 1.0):
        """ Add edge e = (u,v) along with p(e) """
        (u,v) = (min(u,v),max(u,v))
        self.Edges.append((u,v,id))
        self.edict[(u,v,id)] = prob 
        self.notedict[(u,v,id)] = 1-prob 
        self.weights[(u,v,id)] = weight
    
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
        assert ((u,v,id) in self.edict and (u,v,id) in self.notedict)
        self.update_edge_prob(u, v, id, self.get_next_prob(u,v,id,type))

    def update_edge_prob(self, u, v, id, prob):
        """ 
        Updates the probability of an edge to a given value
        """
        if type(prob) is not list: # Deterministic update
            (u,v) = (min(u,v),max(u,v))
            assert ((u,v,id) in self.edict and (u,v,id) in self.notedict)
            self.edict[(u,v,id)] = prob  
            self.notedict[(u,v,id)] = 1 - prob
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
        nbrs = {} # key = vertex id, value = [list of edge ids]
        for e in self.Edges:
            u,v,_id = e
            _tmp = nbrs.get(u,[])
            _tmp.append((v,_id))
            nbrs[u] = _tmp 
            _tmp = nbrs.get(v,[])
            _tmp.append((u,_id))
            nbrs[v] = _tmp 
        return nbrs 
    
    def bfs_sample(self,source,target, seed = 1):
        """ For Reachability query. """
        # print('bfs_sample multigraph')
        # print(self.Edges)
        nbrs = self.construct_nbrs() # Constructs node => Nbr incidence dictionary. 
        queue = [source]
        reached_target = 0
        sample = []
        prob_sample = 1.0
        random.seed(seed)
        visited = {}
        while len(queue) and reached_target == 0: # MC+BFS loop
            u = queue.pop(0)
            # print('pop: ',u)
            # visited[u] = True
            if u == target:
                reached_target = 1
                break
            for v, eid in nbrs.get(u,[]):
                # print('traversing ',v,' via ',eid)
                (uu,vv) = (min(u,v),max(u,v))
                p = self.edict.get((uu,vv,eid),-1)
                if p == -1: #
                    # print(sample,'\n',(uu,vv),' ',u) 
                    continue 
                if random.random() < p:
                    if (not visited.get(eid,False)):
                        visited[eid] = True
                        sample.append((uu,vv))
                        # print('added ',(uu,vv),' into poss world')
                        prob_sample *= p 
                        queue.append(v)
                        if v == target:
                            reached_target = 1
                            break 
                else:
                    prob_sample *= (1-p)
                # print(visited)
        support_value = reached_target
        # print(source,target,sample,support_value)
        return sample, prob_sample,support_value # possible world G, Pr(G), Reach/Not

    def dijkstra_sample(self,source,target, seed = 1):
        """ For SP query (unweighted graph). """
        # print(self.Edges)
        nbrs = self.construct_nbrs() # Constructs node => [Nbr nodes list] incidence dictionary. 
        reached_target = 0
        sample = []
        prob_sample = 1.0
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
            # visited[u] = True
            if u == target:
                reached_target = 1
                break
            for v,eid in nbrs.get(u,[]):
                (uu,vv) = (min(u,v),max(u,v))
                dist_uv = dists[u] + self.weights[(uu,vv,eid)]
                p = self.edict.get((uu,vv,eid),-1)
                if p == -1: # unexpected edge.
                    # print(sample,'\n',(uu,vv),' ',u) 
                    continue 
                if random.random() < p:
                    if (v not in seen) or (dist_uv < seen[v]):
                        seen[v] = dist_uv
                        sample.append((uu,vv))
                        prob_sample *= p 
                        heappush(heap,(dist_uv,v))
                        if v == target:
                            reached_target = 1
                            dists[v] = dist_uv
                            break 
                else:
                    prob_sample *= (1-p)
        support_value = dists.get(target,INFINITY)
        # print(source,target,sample,support_value,dists)
        return sample, prob_sample,support_value # possible world G, Pr(G), Reach/Not
