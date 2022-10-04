

import networkx as nx
import math
from src.query import Query
from src.graph import UGraph

T = 32 # Number of Monte-Carlo sample of possible worlds
def compute_reach(s, t, d):
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
    # for g in probGraph.get_Ksample(T):
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
    num=compute_ub((u,v), s, t, d)
    return num/prob

#compute UB on the numerator of P(e)
def compute_ub(e, s, t, d):
    ub=0.0
    path_l=list(nx.all_simple_paths(G, s, t, cutoff=d))
    paths=path_l.copy()
    Je=[]
    save_prob={} 
    i=0
    sum_pr=0
    for path in map(nx.utils.pairwise, path_l):
        p=(list(path))
        pr_j=compute_prob_path(p)
        save_prob[i]=pr_j
        i+=1
        if e in p:
            Je.append(p)
            sum_pr+=compute_prob_path(p)
    pr_j=0
    Rstd_G=compute_reach(s,t,d)
    for p in Je:
        pr_j=compute_prob_path(list(p))
        ub+=pr_j*(1-(Rstd_G-sum_pr))

    return ub

#tuple((e[0], e[1]))

def find_e(G, s, t, d, e_clean ): #Algorithm 1 of the paper
    NL=[]
    DL=[]  
    for i in range(len(e_clean)):
        U_e=compute_ub(e_clean[i], s, t, d)
        NL.append((e_clean[i][0],e_clean[i][1], 1.0, U_e))
    
    #NL=sorted(NL.items(), key=lambda item: item[1], reverse=True)
    NL.sort(key=lambda a: a[3])    
    e_clean.sort(key=lambda a: a[3])
    DL=e_clean.copy()
    Pmax=-math.inf
    e=(-1,-1, -1.0, -1.0)
    f=True
    while f==True:
        en=NL.pop(0)
        print('en: ',en)
        ed=DL.pop(0)
        Pstar_en=compute_Pstar(en)
        if Pstar_en>Pmax:
            e=en
            Pmax=Pstar_en
            
        Pstar_ed=compute_Pstar(ed)
        if Pstar_ed>Pmax:
            e=ed
            Pmax=Pstar_ed
            
        if en[1]/ed[3]<Pmax: 
            f=False
        
    return e
        
    



 
G=nx.read_edgelist("RelComp/test_wgraph.txt", delimiter=" ", create_using=nx.Graph(), nodetype=int, data=(("weight", float), ("prob", float)))
print(G.edges)
# e_clean=[(3, 0, 1.0, 0.5),
#          (1, 2, 1.0, 0.6),
#          (1, 3, 1.0, 0.3)] #Edges to clean
# e_clean = [
#     (0, 4, 1.0, 1.0),
#     (1, 2, 1.0, 0.25),
#     (2, 0, 1.0, 0.75),
#     (2, 6, 1.0, 0.75),
#     (3, 4, 1.0, 0.5),
#     (5, 1, 1.0, 0.5),
#     (6, 5, 1.0, 0.5),
#     (6, 1, 1.0, 0.75),
#     (6, 4, 1.0, 0.75)
# ]
#e_clean=G.edges(data=True)
e_clean = [(e[0],e[1],e[2]['weight'],e[2]['prob']) for e in G.edges(data=True)]
print(e_clean)
s=3
t=2
d=3


e_star=find_e(G, s, t, d, e_clean)
print("Execution completed e*: (",e_star[0],", ",e_star[1], ")")
print(e_star)


