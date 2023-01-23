import networkx as nx
import math
from src.query import Query
from src.graph import UGraph
from src.utils import *
from matplotlib import pyplot as plt
import argparse 
from copy import deepcopy
parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, default="default")
parser.add_argument("-T",'--T',type = int, default = 10, help= '#of Possible worlds in MC sampling')
parser.add_argument("-v", "--verbose", action='store_true')
parser.add_argument('-s','--source',type = str, default = None)
parser.add_argument('-t','--target',type = str, default = None)
parser.add_argument("-maxd", "--maxhop", type=int, default=1)
parser.add_argument("-b", "--budget", type=int, default=1)


args = parser.parse_args()
s = args.source
t = args.target
d = args.maxhop
budget = args.budget
p_min , p_max = 0.2,0.75 # According to the paper

def compute_reach(s, t, d, probGraph = None): # According to paper it shouldn't compute reach exactly, rather using monte-carlo method.
    if probGraph is None: 
        probGraph = UGraph()
        for e in G.edges:
            probGraph.add_edge(e[0],e[1],prob=G[e[0]][e[1]]['prob'])
    # Q = Query(probGraph, qtype='reach',args = {'u':s,'v':t})
    Q = Query(probGraph, qtype='reach_d',args = {'u':s,'v':t, 'd':d})
    
    # Exact Computation
    Q.eval()
    r = Q.freq_distr[1]
    # Approximate computation
    # r=0.0
    # for g in probGraph.get_Ksample(args.T):
    #     omega = Q.evalG(g[0])
    #     if omega == 1:
    #         r+= g[1]
    return r

def compute_prob_path(p):
    prob=1
    for u,v in p:
        prob*=G[u][v]['prob']
    return prob

def compute_Pstar(e): #approximation of P*(e) using the upper bound 
    u=e[0]
    v=e[1]
    prob=G[u][v]['prob']
    num=e[3]
    return num/prob

#compute UB on the numerator of P(e)
def compute_ub(e, path_l, Rstd_G):
    Je=[]
    sum_pr=0
    for path in map(nx.utils.pairwise, path_l):
        p=(list(path))
        if ((e[0], e[1]) in p) or ((e[1], e[0]) in p):
            Je.append(p)
            sum_pr+=compute_prob_path(p)
    #pr_j=0
    ub=sum_pr*(1-(Rstd_G-sum_pr))

    return ub



def find_e(G, s, t, d, e_clean, probGraph= None): #Algorithm 1 of the paper
    NL=[]
    DL=[] 
    path_l=list(nx.all_simple_paths(G, s, t, cutoff=d)) 
    Rstd_G=compute_reach(s,t,d, probGraph=probGraph)
    for i in range(len(e_clean)):
        U_e=compute_ub(e_clean[i], path_l, Rstd_G)
        NL.append((e_clean[i][0],e_clean[i][1], 1.0, U_e))
    
    NL.sort(key=lambda a: a[3], reverse=True)    
    #print(NL)
    e_clean.sort(key=lambda a: a[3])
    DL=e_clean.copy()
    #print(DL)
    Pmax=-math.inf
    e=(-1,-1, -1.0, -1.0)
    f=True

    while f==True:
        en=NL.pop(0)
        ed=DL.pop(0)
        Pstar_en=compute_Pstar(en)
        if Pstar_en>Pmax:
            e=en
            Pmax=Pstar_en  
        for en1 in NL:
            if (en1[0]==ed[0] and en1[1]==ed[1]) or (en1[0]==ed[1] and en1[1]==ed[0]):
                ed=en1         
        Pstar_ed=compute_Pstar(ed)
        if Pstar_ed>Pmax:
            e=ed
            Pmax=Pstar_ed
            
        if en[3]/ed[3]<Pmax: 
            f=False
        
    return e
        
def doCleanto0(e_star,probGraph):
    probGraph.update_edge_prob(u = e_star[0],v = e_star[1], prob = 0)
def doCleanto1(e_star,probGraph):
    probGraph.update_edge_prob(u = e_star[0],v = e_star[1], prob = 1)

def compute_uncertainty(probGraph,s,t,d, aftercleaning = False, estar = None):
    if aftercleaning:
        # clean the single-best edge returned by baseline
        probGraph_copy = deepcopy(probGraph)

        doCleanto0(estar,probGraph)
        Q = Query(probGraph, qtype='reach_d',args = {'u':s,'v':t, 'd':d})
        Q.eval()
        en0 = Q.compute_entropy()

        doCleanto1(e_star,probGraph_copy)
        Q = Query(probGraph_copy, qtype='reach_d',args = {'u':s,'v':t, 'd':d})
        Q.eval()
        en1 = Q.compute_entropy()
        expected_entropy = (0.5 * en0 + (1-0.5) * en1)
        return expected_entropy
    else:
        Q = Query(probGraph, qtype='reach_d',args = {'u':s,'v':t, 'd':d})
        Q.eval()
        return Q.compute_entropy()

if __name__=='__main__': 
    if args.dataset == 'test':
        G=nx.read_edgelist("test_graph_TKDE.txt", delimiter=" ", create_using=nx.Graph(), nodetype=int, data=(("weight", float), ("prob", float)))
        s,t = int(s),int(t)
    else:
        probGraph = get_dataset(args.dataset)
        G = probGraph.get_weighted_graph_rep()
    # print(G.edges)

    # e_clean=[(3, 0, 1.0, 0.5),
    #           (0, 2, 1.0, 0.7),
    #           (1, 3, 1.0, 0.3)] #Edges to clean


    e_clean = [(e[0],e[1],e[2]['weight'],e[2]['prob']) for e in G.edges(data=True) if e[2]['prob']<= p_max and e[2]['prob']>=p_min]
    if args.verbose: print("Candidate edges: ", e_clean)
    # s=3
    # t=2
    # s = 2
    # t = 3
    # d=3
    # budget=2
    if args.verbose: print('input: source = ',s,' target = ',t,' #hops <= ',d,' budget: ',budget)
    # print(G.nodes())

    #pruning strategy
    #pruning by reverse shortest path
    for v in G.nodes():
        ds_v=nx.dijkstra_path_length(G, v, s, weight='weight')
        dt_v=nx.dijkstra_path_length(G, v, t, weight='weight')
        e_copy=e_clean.copy()
        if ds_v+dt_v>d:
            for a, b, w, p in e_copy:
                if a==v or b==v:
                    e_clean.remove((a,b,w,p))
                    

    if args.verbose: print("Candidate edges after pruning: ", e_clean)
    e_star=find_e(G, s, t, d, e_clean)
    if args.verbose:
        print("Execution completed e*: (",e_star[0],", ",e_star[1], ")")
        nx.draw(G,with_labels = True)
        plt.savefig('graph_TKDE.png')
    print('Uncertainty before cleaning: ',compute_uncertainty(probGraph,s,t,d))
    print('Uncertainty after cleaning: ',compute_uncertainty(probGraph,s,t,d,aftercleaning=True,estar=e_star))
# python AlgoTKDE17.py -d test -s 2 -t 3 -maxd 3 -b 2