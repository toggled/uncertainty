import itertools,os
import networkx as nx
from matplotlib import pyplot as plt 
import random 
from src.graph import UGraph as Graph
from src.graph import UMultiGraph as multiGraph
from networkx.algorithms.bipartite.matching import INFINITY
from scipy.stats import entropy
import math 
from time import time 
pdf_path = 'figs/'
font_dict = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 14,
        }

class Query:
    """
    A generic Query class that exactly evaluating the queries considered in the paper.
    """
    def __init__(self, prob_graph, qtype, args = {}):
        """ Input: 
            An uncertain graph and 
            a query type => ( e.g. Degree, Shortest path length , #Triangles)
        """
        self.p_graph = prob_graph
        assert (isinstance(self.p_graph, Graph))
        self.qtype = qtype
        self.results = [] # contains list of (a possible world G, property-value P(G), prob (Pr[G]) ) triplets.
        self.freq_distr = {} # Pr[Omega = Omega_i]
        self.plot_properties = {} 
        self.possible_world_statistic = {}
        self.support_set = set()
        self.confidence = {}
        self.evaluation_times = []
        if self.qtype == 'reach' or self.qtype == 'sp':
            self.u = args['u']
            self.v = args['v']
        if self.qtype == 'reach_d':
            self.u = args['u']
            self.v = args['v']
            self.d = args['d']
    
    def reset(self, prob_graph):
        """ Reset the query with a new uncertain graph """
        self.p_graph = prob_graph
        assert (isinstance(self.p_graph, Graph))
        self.results = [] # contains list of (a possible world G, property-value P(G), prob (Pr[G]) ) triplets.
        self.freq_distr = {} 
        self.possible_world_statistic = {}
        self.support_set = set()
        self.confidence = {}
        self.evaluation_times = []
    
    def clear(self):
        self.p_graph = None
        self.results = None
        self.freq_distr = None 
        self.possible_world_statistic = None
        self.support_set = None
        self.confidence = None
        self.evaluation_times = None

    # @profile
    def eval(self,dontenumerateworlds=True):
        """
        Given vertices u,v,w (some possibly None) as input, evaluates the query and 
        computes (world, value_of_the_property, prob) triplet for all possible worlds.
        """
        #         if self.qtype == 'degree':
        #             self.plot_properties['xlabel'] = 'Degree(\"' + str(u)+'\")'
        #             self.plot_properties['ylabel'] = 'Prob.'
        #             assert (u!=None)
        #             for x, G in enumerate(self.p_graph.enumerate_worlds()):
        #                 deg = 0
        #                 for edge in G[0]:
        #                     # print(edge)
        #                     deg += [0,1][u in edge]
        #                 self.results.append((G[0], deg, G[1]))
        #                 self.freq_distr[deg] = self.freq_distr.get(deg,0) + G[1] 
        #                 self.possible_world_statistic[deg] = self.possible_world_statistic.get(deg,0) + 1
        #                 self.support_set.add(deg)
        # #                 self.confidence[deg] = self.confidence.get(deg, 0) + G[1]
        if self.qtype == 'reach': # Reachability
                u = self.u 
                v = self.v
                assert (u != None and v != None)
                self.plot_properties['xlabel'] = 'Reachability ('+str(u)+'~'+str(v)+')'
                self.plot_properties['ylabel'] = 'Prob.'
                for x, G in enumerate(self.p_graph.enumerate_worlds()):
                    start_tm = time()
                    nx_G = nx.Graph()
                    nx_G.add_edges_from(G[0])
                    reachable = 0
                    if (u in nx_G) and (v in nx_G):
                        if (nx.has_path(nx_G,u,v)):
                            reachable = 1 
                    if dontenumerateworlds:
                        self.results.append((None, reachable, G[1]))
                    else:
                        self.results.append((G[0], reachable, G[1]))
                    self.freq_distr[reachable] = self.freq_distr.get(reachable,0) + G[1] 
                    self.possible_world_statistic[reachable] = self.possible_world_statistic.get(reachable,0) + 1
                    self.support_set.add(reachable)
                    self.evaluation_times.append(time()-start_tm)

        if self.qtype == 'reach_d': # Reachability
                u = self.u 
                v = self.v
                d = self.d
                assert (u != None and v != None)
                self.plot_properties['xlabel'] = 'Reachability ('+str(u)+'~'+str(v)+'<='+str(d)+')'
                self.plot_properties['ylabel'] = 'Prob.'
                for x, G in enumerate(self.p_graph.enumerate_worlds()):
                    start_tm = time()
                    nx_G = nx.Graph()
                    nx_G.add_edges_from(G[0])
                    reachable = 0
                    if (u in nx_G) and (v in nx_G):
                        if (nx.has_path(nx_G,u,v)):
                            if nx.shortest_path_length(nx_G, source=u, target=v) <= d:
                                reachable = 1 
                    if dontenumerateworlds:
                        self.results.append((None, reachable, G[1]))
                    else:
                        self.results.append((G[0], reachable, G[1]))
                    self.freq_distr[reachable] = self.freq_distr.get(reachable,0) + G[1] 
                    self.possible_world_statistic[reachable] = self.possible_world_statistic.get(reachable,0) + 1
                    self.support_set.add(reachable)
                    self.evaluation_times.append(time()-start_tm)

        if self.qtype == 'sp': # length of shortest path
                print('sp')
                u = self.u 
                v = self.v
                assert (u != None and v != None)
                self.plot_properties['xlabel'] = 'Reachability ('+str(u)+'~'+str(v)+')'
                self.plot_properties['ylabel'] = 'Prob.'
                for x, G in enumerate(self.p_graph.enumerate_worlds()):
                    # print(G)
                    start_tm = time()
                    nx_G = nx.Graph()
                    nx_G.add_edges_from(G[0])
                    sp_len = INFINITY
                    if (u in nx_G) and (v in nx_G):
                        if (nx.has_path(nx_G,u,v)):
                            sp_len = nx.shortest_path_length(nx_G, source=u, target=v)
                    if dontenumerateworlds:
                        self.results.append((None, sp_len, G[1]))
                    else:
                        self.results.append((G[0], sp_len, G[1]))
                    self.freq_distr[sp_len] = self.freq_distr.get(sp_len,0) + G[1] 
                    self.possible_world_statistic[sp_len] = self.possible_world_statistic.get(sp_len,0) + 1
                    self.support_set.add(sp_len)
                    self.evaluation_times.append(time()-start_tm)

        if self.qtype =="diam":
            self.plot_properties['xlabel'] = 'Diam'
            self.plot_properties['ylabel'] = 'Prob.'
            for x, G in enumerate(self.p_graph.enumerate_worlds()):
                    start_tm = time()
                    nx_G = nx.Graph()
                    nx_G.add_edges_from(G[0])
                    
                    if nx_G.number_of_edges() == 0:
                        diam = INFINITY
                    else:
                        if not nx.is_connected(nx_G):
                            diam = INFINITY
                        else:
                            diam = nx.diameter(nx_G)
                    if dontenumerateworlds:
                        self.results.append((None, diam, G[1]))
                    else:
                        self.results.append((G[0], diam, G[1]))
                    self.freq_distr[diam] = self.freq_distr.get(diam,0) + G[1] 
                    self.possible_world_statistic[diam] = self.possible_world_statistic.get(diam,0) + 1
                    self.support_set.add(diam)
                    self.evaluation_times.append(time()-start_tm)
            
        if self.qtype == 'tri':
            self.plot_properties['xlabel'] = '#Triangles'
            self.plot_properties['ylabel'] = 'Prob.'
            
            for x, G in enumerate(self.p_graph.enumerate_worlds()):
                start_tm = time()
                g = nx.Graph()
                g.add_edges_from(G[0])
                num_triangles = sum(nx.triangles(g).values()) / 3
                if dontenumerateworlds:
                    self.results.append((None, num_triangles, G[1]))
                else:
                    self.results.append((G[0], num_triangles, G[1]))
                self.freq_distr[num_triangles] = self.freq_distr.get(num_triangles,0) + G[1] 
                self.possible_world_statistic[num_triangles] = self.possible_world_statistic.get(num_triangles,0) + 1
                self.support_set.add(num_triangles)
                self.evaluation_times.append(time()-start_tm)
#                 self.confidence[num_triangles] = self.confidence.get(num_triangles, 0) + G[1]
                
        # if self.qtype == 'shortest_path_len':
        #     self.plot_properties['xlabel'] = 'Len_SP(' + str(u)+'~'+str(v)+')'
        #     self.plot_properties['ylabel'] = 'Prob.'
        #     assert (u!=None and v!=None)
        #     for x, G in enumerate(self.p_graph.enumerate_worlds()):
        #         g = nx.Graph()
        #         g.add_edges_from(G[0])
        #         length = None
        #         if (u not in g) or (v not in g):
        #             length = INFINITY
        #         else:
        #             try:
        #                 length = nx.shortest_path_length(g,u,v)
        #             except:
        #                 length = INFINITY
        #         self.results.append((G[0],length, G[1]))
        #         self.freq_distr[length] = self.freq_distr.get(length,0) + G[1] 
        #         self.possible_world_statistic[length] = self.possible_world_statistic.get(length,0) + 1
        #         self.support_set.add(length)

    def evalG(self,G):
        """ Function to evaluate a given possible world G"""
        assert isinstance(G,list)
        if self.qtype == 'reach': # Reachability
            u = self.u 
            v = self.v
            assert (u != None and v != None)
            nx_G = nx.Graph()
            nx_G.add_edges_from(G)
            reachable = 0
            if (u in nx_G) and (v in nx_G):
                if (nx.has_path(nx_G,u,v)):
                    reachable = 1 

            return reachable

        if self.qtype == 'sp': # length of shortest path
            u = self.u 
            v = self.v
            assert (u != None and v != None)
            
            nx_G = nx.Graph()
            nx_G.add_edges_from(G)
            sp_len = INFINITY
            if (u in nx_G) and (v in nx_G):
                if (nx.has_path(nx_G,u,v)):
                    sp_len = nx.shortest_path_length(nx_G, source=u, target=v)
            return sp_len

        if self.qtype =="diam":
            self.plot_properties['xlabel'] = 'Diam'
            self.plot_properties['ylabel'] = 'Prob.'
           
            nx_G = nx.Graph()
            nx_G.add_edges_from(G)
            if nx_G.number_of_edges() == 0:
                diam = INFINITY
            else:
                if not nx.is_connected(nx_G):
                    diam = INFINITY
                else:
                    diam = nx.diameter(nx_G)
            
            return diam 
            
        if self.qtype == 'tri':
            g = nx.Graph()
            g.add_edges_from(G)
            num_triangles = sum(nx.triangles(g).values()) / 3
            return num_triangles

    def compute_entropy(self, base = 2):
        """ Given a base for logarithm, returns the entropy of the Property_value distribution. """
        if str(base) == 'e':
            return entropy([j for i,j in self.get_distribution()])
        return entropy([j for i,j in self.get_distribution()], base = base)
    
    def get_distribution(self):
        """
        Returns the distribution of property values over the possible worlds.
        """
        return sorted([(i,j) for (i,j) in self.freq_distr.items()])
    
    def distr_plot(self, saveplot = False, add_text = {'x': 2, 'y': 0.65, 'text': None, 'font': font_dict}, label = 'Conf', filename = None):
        """
        Plots the distribution of property values.
        """
        pencolor = 'k' #black/red/blue etc
        penstyle = 'solid' # strock-style: solid/dashed etc.
        assert (len(self.freq_distr)!=0)
        keys = sorted(self.freq_distr.keys())
        plt.bar([str(i) for i in keys],[self.freq_distr[i] for i in keys], label = label, width = 0.1, color = pencolor, ls = penstyle)
        plt.ylim((0,1))
        plt.xlabel(self.plot_properties['xlabel'])
        plt.ylabel(self.plot_properties['ylabel'])
        plt.legend()
        
        if add_text['text'] is not None:
            plt.text(add_text['x'],add_text['y'], add_text['text'], fontdict = add_text['font'])
            
        if saveplot:
            os.system('mkdir -p '+pdf_path)
            if filename is None:
                filename = self.qtype+'.pdf'
            plt.savefig(pdf_path+filename)
        plt.show()
    
    def print_results(self):
        """ Prints the (possible world,property value in that world, world probability) triplets."""
        assert (len(self.results)!=0)
        print("Worlds, Prob: ")
        for res in self.results:
            print(res)
            
    def get_expectation(self):
        """
        Compute Expected value of the (property_value, prob) pairs
        """
        return sum([x*y for x,y in self.freq_distr.items()])
    
    def get_support(self):
        """ 
        Returns the support set. 
        """
        return self.support_set
    
    
    def get_conf(self, prop_val):
        """ 
        Given a property value, returns its confidence.
        Conf() is the same as frequency distribution of property value. 
        """
        assert prop_val in self.freq_distr
        return self.freq_distr[prop_val]
    
    def get_entropy(self, base = 2):
        """
        Computes entropy of the distribution of property values.
        """
        print("Entropy: ", self.compute_entropy(base = base))
    
    def plot_possible_worldstat(self, saveplot = False):
        """
        Computes the frequency count of the possible worlds having the same property values.
        """
        keys = sorted(self.possible_world_statistic.keys())
        plt.bar([str(i) for i in keys],[self.possible_world_statistic[i] for i in keys],width = 0.1)
        plt.xlabel(self.plot_properties['xlabel'])
        plt.ylabel('#worlds')
        if saveplot:
            os.system('mkdir -p '+pdf_path)
            plt.savefig(pdf_path+self.qtype + '(worldcounts).pdf', bbox_inches = 'tight')
        plt.show()

    # def constructTables(self, op='o1'):
    #     """ 
    #     Constructs C, DeltaC and phi Inverse 
    #     """
    #     # TODO: Implement for diam and triangle query as well. 
    #     self.C = {} 
    #     self.DeltaC = {}
    #     self.phiInv = {} 
    #     if self.qtype == 'reach': # Reachability
    #         for i,G in enumerate(self.p_graph.enumerate_worlds2()):
    #             Pr_G = G[1] # Pr(G_i)
    #             sum_Pr_e = G[2] # \sum Pr(e) for all e \in G_i
    #             self.C[i] = {}
    #             self.DeltaC[i] = {}
    #             # Constructing C[] and DeltaC[]
    #             if len(G[0]) == 0: # Empty Graph
    #                 # let us denote empty edge as (-1,-1)
    #                 e = (-1,-1)
    #                 self.C[i][e] = Pr_G 
    #                 self.DeltaC[i][e] = 0
    #             else:
    #                 for e in G[0]:
    #                     pe = self.p_graph.edict[e] #p(e)
    #                     self.C[i][e] = pe * Pr_G/sum_Pr_e # p(e) * Pr(G_i) / sum (p(e))
    #                     next_pe = self.p_graph.get_next_prob(e[0],e[1],op) # p_up(e)
    #                     Del_pe = (pe - next_pe)
    #                     if next_pe != 0:
    #                         times_pe = (pe/next_pe)
    #                         C_up = (next_pe * (Pr_G /times_pe))/(sum_Pr_e - Del_pe) # p_up(e) * Pr_up (G_i)/ sum (p_up(e))
    #                         self.DeltaC[i][e] = C_up - self.C[i][e]
    #                     else:
    #                         C_up  = 0 # If p_up(e) = 0, C_up[i][e] = 0, 
    #                         self.DeltaC[i][e] = C_up - self.C[i][e]
                
    #             # Construct phiInv
    #             u = self.u 
    #             v = self.v
    #             assert (u != None and v != None)
    #             nx_G = nx.Graph()
    #             nx_G.add_edges_from(G[0])
    #             reachable = 0
    #             if (u in nx_G) and (v in nx_G):
    #                 if (nx.has_path(nx_G,u,v)):
    #                     reachable = 1
    #             if reachable not in self.phiInv:
    #                 self.phiInv[reachable] = [i]
    #             else:
    #                 self.phiInv[reachable].append(i)
    #     if self.qtype == 'diam':
    #         raise Exception("Not implemented yet")
        
    #     if self.qtype == "num_triangles":
    #         raise Exception("Not implemented yet")

    # def updateTables(self,op='o1'):
    #     """ 
    #     Updates C, DeltaC and phi Inverse after probability update
    #     """
    #      # TODO: Implement for diam and triangle query as well. 
    #     if self.qtype == 'reach': # Reachability
    #         for i,G in enumerate(self.p_graph.enumerate_worlds2()):
    #             Pr_G = G[1]
    #             sum_Pr_e = G[2]
    #             if len(G[0]) == 0: # Empty Graph
    #                 # let us denote empty edge as (-1,-1)
    #                 e = (-1,-1)
    #                 self.C[i][e] = Pr_G 
    #                 self.DeltaC[i][e] = 0
    #             else:
    #                 for e in G[0]:
    #                     pe = self.p_graph.edict[e]
    #                     self.C[i][e] = pe * Pr_G/sum_Pr_e 
    #                     next_pe = self.p_graph.get_next_prob(e[0],e[1],op)
    #                     Del_pe = (pe - next_pe)
    #                     if next_pe != 0:
    #                         times_pe = (pe/next_pe)
    #                         C_up = (next_pe * (Pr_G /times_pe))/(sum_Pr_e - Del_pe) # p_up(e) * Pr_up (G_i)/ sum (p_up(e))
    #                         self.DeltaC[i][e] = C_up - self.C[i][e]
    #                     else:
    #                         C_up  = 0 # If p_up(e) = 0, C_up[i][e] = 0
    #                         self.DeltaC[i][e] = C_up - self.C[i][e]

    #     if self.qtype == 'diam':
    #         raise Exception("Not implemented yet")
        
    #     if self.qtype == "num_triangles":
    #         raise Exception("Not implemented yet")

    def constructTables(self, op='o1', verbose = False):
        """ 
        Constructs C, DeltaC and phi Inverse 
        """
        self.PrG = {} # world i => Pr(G_i) for all possible world G_i
        self.index = {} 
        self.phiInv = {} # property value Omega_i => worlds where omega_i is observed.
        if (verbose):
            self.G = {} 
        if self.qtype == 'reach': # Reachability
            for i,G in enumerate(self.p_graph.enumerate_worlds()):
                Pr_G = G[1] # Pr(G_i)
                self.PrG[i] = Pr_G
                self.index[i] = {}
                for e in G[0]:
                    self.index[i][e] = 1
                # Construct phiInv
                u = self.u 
                v = self.v
                assert (u != None and v != None)
                nx_G = nx.Graph()
                nx_G.add_edges_from(G[0])
                reachable = 0
                if (u in nx_G) and (v in nx_G):
                    if (nx.has_path(nx_G,u,v)):
                        reachable = 1
                if reachable not in self.phiInv:
                    self.phiInv[reachable] = [i]
                else:
                    self.phiInv[reachable].append(i)
        if self.qtype == 'diam':
            for i,G in enumerate(self.p_graph.enumerate_worlds()):
                Pr_G = G[1] # Pr(G_i)
                self.PrG[i] = Pr_G
                self.index[i] = {}
                for e in G[0]:
                    self.index[i][e] = 1
                
                # Construct phiInv
                if (len(G[0])) == 0:
                    diam = 0 
                else:
                    nx_G = nx.Graph()
                    nx_G.add_edges_from(G[0])
                    if not nx.is_connected(nx_G):
                        diam = math.inf
                    else:
                        diam = nx.diameter(nx_G)
                    if diam not in self.phiInv:
                        self.phiInv[diam] = [i]
                    else:
                        self.phiInv[diam].append(i)  
        
        if self.qtype == "tri":
            # print('Tri')
            for i,G in enumerate(self.p_graph.enumerate_worlds()):
                if (verbose):
                    self.G[i] = G[0] 
                Pr_G = G[1] # Pr(G_i)
                self.PrG[i] = Pr_G
                self.index[i] = {}
                for e in G[0]:
                    self.index[i][e] = 1
                
                # Construct phiInv
                nx_G = nx.Graph()
                nx_G.add_edges_from(G[0])
                num_triangles = sum(nx.triangles(nx_G).values()) / 3
                if num_triangles not in self.phiInv:
                    self.phiInv[num_triangles] = [i]
                else:
                    self.phiInv[num_triangles].append(i)

    def constructTables_S(self, op = 'o1', K = 100, verbose = False):
        """ Sampling variant of the auxiliary data structure construction """
        self.PrG = {} # world i => Pr(G_i) for all possible world G_i
        self.index = {} 
        self.phiInv = {} # property value Omega_i => worlds where omega_i is observed.
        if (verbose):
            self.G = {} 
        if self.qtype == 'reach': # Reachability
            for i,G in enumerate(self.p_graph.get_Ksample(K)):
                Pr_G = G[1] # Pr(G_i)
                self.PrG[i] = Pr_G
                self.index[i] = {}
                for e in G[0]:
                    self.index[i][e] = 1
                # Construct phiInv
                u = self.u 
                v = self.v
                assert (u != None and v != None)
                nx_G = nx.Graph()
                nx_G.add_edges_from(G[0])
                reachable = 0
                if (u in nx_G) and (v in nx_G):
                    if (nx.has_path(nx_G,u,v)):
                        reachable = 1
                if reachable not in self.phiInv:
                    self.phiInv[reachable] = [i]
                else:
                    self.phiInv[reachable].append(i)
        if self.qtype == 'diam':
            for i,G in enumerate(self.p_graph.get_Ksample(K)):
                Pr_G = G[1] # Pr(G_i)
                self.PrG[i] = Pr_G
                self.index[i] = {}
                for e in G[0]:
                    self.index[i][e] = 1
                
                # Construct phiInv
                if (len(G[0])) == 0:
                    diam = 0 
                else:
                    nx_G = nx.Graph()
                    nx_G.add_edges_from(G[0])
                    if not nx.is_connected(nx_G):
                        diam = math.inf
                    else:
                        diam = nx.diameter(nx_G)
                    if diam not in self.phiInv:
                        self.phiInv[diam] = [i]
                    else:
                        self.phiInv[diam].append(i)
        
        if self.qtype == "tri":
            # print('Tri')
            for i,G in enumerate(self.p_graph.get_Ksample(K)):
                if (verbose):
                    self.G[i] = G[0] 
                Pr_G = G[1] # Pr(G_i)
                self.PrG[i] = Pr_G
                self.index[i] = {}
                for e in G[0]:
                    self.index[i][e] = 1
                
                # Construct phiInv
                nx_G = nx.Graph()
                nx_G.add_edges_from(G[0])
                num_triangles = sum(nx.triangles(nx_G).values()) / 3
                if num_triangles not in self.phiInv:
                    self.phiInv[num_triangles] = [i]
                else:
                    self.phiInv[num_triangles].append(i)

    def updateTables(self, ej, op='o1'):
        """ 
        Updates C, DeltaC and phi Inverse after probability update
        """
        p_up_e = self.p_graph.get_next_prob(ej[0],ej[1],op)
        for i in self.index:
            if ej in self.index[i]:
                self.PrG[i] = self.PrG[i] * p_up_e / self.p_graph.edict[ej]
            else:
                self.PrG[i] = self.PrG[i] * (1-p_up_e) / (1-self.p_graph.edict[ej])

class wQuery(Query):
    """
    A generic Query class that exactly evaluates the queries on a weighted graph.
    """
    def __init__(self, prob_graph, qtype, args = {}):
        """ Input: 
            An uncertain graph and 
            a query type => ( e.g. Degree, Shortest path length , #Triangles)
        """
        super().__init__(prob_graph,qtype,args)

    def evalG(self,G):
        """ Function to evaluate a given possible world G (weighted)"""
        assert isinstance(G,list)
        if self.qtype == 'reach': # Reachability
            return super().evalG(G)

        if self.qtype == 'sp': # length of shortest path
            u = self.u 
            v = self.v
            assert (u != None and v != None)
            
            nx_G = nx.Graph()
            nx_G.add_weighted_edges_from([(e[0],e[1], self.p_graph.weights[(e[0],e[1])]) for e in G])
            sp_len = INFINITY
            if (u in nx_G) and (v in nx_G):
                if (nx.has_path(nx_G,u,v)):
                    sp_len = nx.dijkstra_path_length(nx_G, source=u, target=v)
            return sp_len

        if self.qtype =="diam":
            return super().evalG(G) 
            
        if self.qtype == 'tri':
            return super().evalG(G)

class multiGraphQuery(Query):
    """
    A generic Query class that exactly evaluates the queries on an unweighted multigraph.
    """
    def __init__(self, prob_graph, qtype, args = {}):
        """ Input: 
            An uncertain graph and 
            a query type => ( e.g. Degree, Shortest path length , #Triangles)
        """
        assert isinstance(prob_graph,multiGraph)
        super().__init__(prob_graph,qtype,args)

    def evalG(self,G):
        """ Function to evaluate a given possible world G (multigraph)"""
        # print('Multigraph eval.')
        assert isinstance(G,list)
        if self.qtype == 'reach': # Reachability
            u = self.u 
            v = self.v
            assert (u != None and v != None)
            nx_G = nx.MultiGraph()
            nx_G.add_edges_from(G)
            reachable = 0
            if (u in nx_G) and (v in nx_G):
                if (nx.has_path(nx_G,u,v)):
                    reachable = 1 

            return reachable

        if self.qtype == 'sp': # length of shortest path
            u = self.u 
            v = self.v
            assert (u != None and v != None)
            
            nx_G = nx.MultiGraph()
            nx_G.add_edges_from(G)
            sp_len = INFINITY
            if (u in nx_G) and (v in nx_G):
                if (nx.has_path(nx_G,u,v)):
                    sp_len = nx.shortest_path_length(nx_G, source=u, target=v)
            return sp_len

        if self.qtype =="diam":
            self.plot_properties['xlabel'] = 'Diam'
            self.plot_properties['ylabel'] = 'Prob.'
           
            nx_G = nx.MultiGraph()
            nx_G.add_edges_from(G)
            if nx_G.number_of_edges() == 0:
                diam = INFINITY
            else:
                if not nx.is_connected(nx_G):
                    diam = INFINITY
                else:
                    diam = nx.diameter(nx_G)
            
            return diam 
            
        if self.qtype == 'tri': # Bug here
            g = nx.MultiGraph()
            g.add_edges_from(G)
            num_triangles = sum(nx.triangles(g).values()) / 3
            return num_triangles

class multiGraphwQuery(multiGraphQuery):
    """
    A generic Query class that exactly evaluates the queries on a weighted multigraph.
    """
    def __init__(self, prob_graph, qtype, args = {}):
        """ Input: 
            An uncertain graph and 
            a query type => ( e.g. reachability, Shortest path length , #Triangles)
        """
        super().__init__(prob_graph,qtype,args)

    def evalG(self,G):
        """ Function to evaluate a given possible world G (weighted multigraph)"""
        # print('Multigraph eval.')
        assert isinstance(G,list)
        if self.qtype == 'reach': # Reachability
            return super().evalG(G)

        if self.qtype == 'sp': # length of shortest path
            # print('weighted multigraph sp query')
            u = self.u 
            v = self.v
            assert (u != None and v != None)
            
            nx_G = nx.MultiGraph()
            nx_G.add_weighted_edges_from([(e[0],e[1], self.p_graph.weights[(e[0],e[1],e[2])]) for e in G])
            sp_len = INFINITY
            if (u in nx_G) and (v in nx_G):
                if (nx.has_path(nx_G,u,v)):
                    sp_len = nx.dijkstra_path_length(nx_G, source=u, target=v)
            return sp_len


        if self.qtype =="diam":
            return super().evalG(G)
            
        if self.qtype == 'tri':
            return super().evalG(G)
