

import networkx as nx
import math
from src.query import Query
from src.graph import UGraph

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



def find_e(G, s, t, d, e_clean ): #Algorithm 1 of the paper
    NL=[]
    DL=[] 
    path_l=list(nx.all_simple_paths(G, s, t, cutoff=d)) 
    Rstd_G=compute_reach(s,t,d)
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
        
    



 
G=nx.read_edgelist("test_graph_TKDE.txt", delimiter=" ", create_using=nx.Graph(), nodetype=int, data=(("weight", float), ("prob", float)))
#print(G.edges)


e_clean=[(3, 0, 1.0, 0.5),
          (0, 2, 1.0, 0.7),
          (1, 3, 1.0, 0.3)] #Edges to clean


#e_clean = [(e[0],e[1],e[2]['weight'],e[2]['prob']) for e in G.edges(data=True)]
print("Candidate edges: ", e_clean)
s=3
t=2
d=3
budget=2



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
                

print("Candidate edges after pruning: ", e_clean)
e_star=find_e(G, s, t, d, e_clean)
print("Execution completed e*: (",e_star[0],", ",e_star[1], ")")


