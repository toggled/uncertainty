import networkx as nx
import math
from time import time
from src.query import Query
from src.graph import UGraph
from src.utils import *
from matplotlib import pyplot as plt
import argparse , os
from copy import deepcopy
import pandas as pd
from src.algorithm import ApproximateAlgorithm
import random
from math import log2
from networkx.exception import NetworkXNoPath
from itertools import combinations

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="default")
parser.add_argument("-T",'--T',type = int, default = 10, help= '#of Possible worlds in MC sampling')
parser.add_argument("-v", "--verbose", action='store_true')
parser.add_argument('-s','--source',type = str, default = None)
parser.add_argument('-t','--target',type = str, default = None)
parser.add_argument("-maxd", "--maxhop", type=int, default=1)
parser.add_argument("-b", "--budget", type=int, default=1)
parser.add_argument("-u",'--utype',type = str, default = 'c2') # c = crowdsourced update, c1=non-adaptive, c2 = adaptive
parser.add_argument('-cr','--cr', type = str, default = 'data/large/crowd/paper_pair.true')
parser.add_argument('-q','--queryf', type=str,help='query file',default = 'data/queries/papers/papers_2.queries') # 'data/queries/ER/ER_15_22_2.queries'
parser.add_argument('-mq','--maxquery',type = int,help='#query pairs to take, maximum = -1 means All queries',default=-1)

args = parser.parse_args()
print(args)

os.environ['time_seed'] = 'True'
opt_N_dict = {
    'default': {'reach': 10},
    'ER_15_22': {'reach': 11, 'sp': 26, 'tri': 6},
    'biomine': {'reach': None},
    'flickr': {'tri': 76},
    'rome': {'sp': 96},
    'papers': {'reach': 71},
    'products': {'reach': 46},
    'restaurants': {'reach': 156}
}
opt_T_dict = {
    'default': {'reach':10},
    'ER_15_22': {'reach': 85, 'sp': 165, 'tri': 100},
    'biomine': {'reach': 10},
    'flickr': {'tri': 51},
    'rome': {'sp': 11},
    'papers': {'reach':10},
    'products': {"reach": 4},
    'restaurants': {'reach': 6}
}
cr_dict = {}
T = opt_T_dict[args.dataset]['reach']
N = opt_N_dict[args.dataset]['reach']
s = args.source
t = args.target
d = args.maxhop
budget = args.budget
ZERO = 10**(-13)
# p_min , p_max = 0.2,0.75 # According to the paper
# h = lambda x: [0,-x*log2(x)][x!=0]
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

def compute_approx_reach(s,t,d,probGraph, numsamples = None):
    random.seed()
    sid = random.randint(0,1000000)
    if numsamples is None:
        numsamples = N*T
    # numsamples = T
    # print('num of samples: ',numsamples)
    func_obj = probGraph.get_Ksample_dhopbfs(K = numsamples,seed=sid,\
                                source=s,target = t, dhop = d, optimiseargs = None)
    hat_Pr = {}
    for _,_, _,omega in func_obj:
        hat_Pr[omega] = hat_Pr.get(omega,0) + (1.0/numsamples)
    # print(hat_Pr)
    reachability = hat_Pr.get(1,0)
    return reachability

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

def quality_Q_st(Rstd_G):
    return h(Rstd_G) + h(1-Rstd_G)

def compute_qual_improve_UQ(G, s,t,d,e_clean,probGraph):
    path_l=list(nx.all_simple_paths(G, s, t, cutoff=d)) 
    Rstd_G = compute_approx_reach(s,t,d,probGraph=probGraph)
    # print('#paths = ',len(path_l))
    # Rstd_G = compute_reach(s,t,d,probGraph)
    # print('Rstd_G = ',Rstd_G)
    UQ = []
    Qstd_G = quality_Q_st(Rstd_G)
    # print('Qstd_G = ',Qstd_G)
    print('len(e_clean) = ', len(e_clean))
    for i in range(len(e_clean)):
        if len(path_l) == 0:
            U_e = 0
        else:
            U_e=compute_ub(e_clean[i], path_l, Rstd_G)
        # print('Ue: ',U_e)
        # print('p_e: ',e_clean[i][3])
        phatstar_e = U_e # page 9, 2nd column
        p_e = e_clean[i][3]
        # print('Rstd_G + (1-p_e)*phatstar_e = ',Rstd_G + (1-p_e)*phatstar_e)
        # print('1-Rstd_G - (1-p_e)*phatstar_e = ', 1-Rstd_G - (1-p_e)*phatstar_e)
        # print('Rstd_G - p_e*phatstar_e = ',Rstd_G - p_e*phatstar_e)
        eqhat = p_e*h(Rstd_G + (1-p_e)*phatstar_e) \
                + p_e*h(1-Rstd_G - (1-p_e)*phatstar_e) \
                + (1-p_e)*h(Rstd_G - p_e*phatstar_e) \
                + (1-p_e)*h(1-Rstd_G+p_e*phatstar_e)
        # Eqhat.append(eqhat)
        UQ.append(Qstd_G - eqhat)
    return UQ 

def find_e(G, s, t, d, e_clean, probGraph= None): #Algorithm 1 of the paper
    NL=[]
    DL=[] 
    path_l=list(nx.all_simple_paths(G, s, t, cutoff=d)) 
    # Rstd_G=compute_reach(s,t,d, probGraph=probGraph)
    Rstd_G = compute_approx_reach(s,t,d,probGraph=probGraph)
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
        
# Single edge selection, single query-pair
def find_e_adaptive(G, s, t, d, inf_set, e_clean, probGraph= None): #Algorithm 2 of the paper (Greedy variant)
    # Single-edge selection among the candidates e_clean
    Uq = compute_qual_improve_UQ(G,s,t,d,e_clean,probGraph)
    descending_uq = sorted(list(zip(e_clean,Uq)),reverse=True, key=lambda x: x[1])
    eqmax = -10000000
    estar = None
    # print('Uq = ',Uq)
    for i in range(len(Uq)): # Simulating the loop over heap, 1st element e,second elemnet Upper bound UQ[e]
        # print(i)
        if descending_uq[i][1]<eqmax: 
            return descending_uq[i][0]
        else:
            e = descending_uq[i][0]  # pop from heap
            e = (e[0],e[1])
            E_deltaQ_st_d = 0
            p_e = probGraph.edict[e]
            for (s,t) in inf_set[e]:
                Q_st_d = compute_approx_reach(s,t,d,probGraph=probGraph)
                # e = descending_uq[i][0]
                pg1 = deepcopy(probGraph)
                doCleanto1(e,pg1)
                R_st_d_c1 = compute_approx_reach(s,t,d,probGraph=pg1)
                doCleanto0(e,pg1)
                R_st_d_c0 = compute_approx_reach(s,t,d,probGraph=pg1)
                Q_st_d_c1 = h(R_st_d_c1) + h(1-R_st_d_c1)
                Q_st_d_c0 = h(R_st_d_c0) + h(1-R_st_d_c0)
                E_deltaQ_st_d += (Q_st_d  - (p_e*Q_st_d_c1 + (1-p_e)*Q_st_d_c0))
            # print('i = ',i, ' E_deltaQ_st_d = ',E_deltaQ_st_d)
            if E_deltaQ_st_d > eqmax:
                eqmax = E_deltaQ_st_d
                estar = e
    return estar

def compute_uncertainty(probGraph,s,t,d, aftercleaning = False, estar = None):
    if aftercleaning:
        # clean the single-best edge returned by baseline
        probGraph_copy = deepcopy(probGraph)

        doCleanto0(estar,probGraph)
        Q = Query(probGraph, qtype='reach_d',args = {'u':s,'v':t, 'd':d})
        Q.eval()
        en0 = Q.compute_entropy()

        doCleanto1(estar,probGraph_copy)
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
        Edgelist = [(e[0],e[1],e[2]['weight'],e[2]['prob']) for e in G.edges(data=True)]
        probGraph = UGraph()
        for (u,v,w,p) in Edgelist:
            probGraph.add_edge(u,v,float(p),weight=float(w))
    else:
        probGraph = get_dataset(args.dataset)
        G = probGraph.get_weighted_graph_rep()
        queries = get_queries(queryfile = args.queryf,maxQ = args.maxquery)
        nodes = G.nodes()
        # print(len(G.nodes), ' ',len(G.edges))
        # import sys
        # sys.exit(1)
    # print(G.edges)

    with open(args.cr,'r') as f:
        for i,line in enumerate(f.readlines()):
            cr_value = int(line.strip())
            cr_dict[probGraph.Edges[i]] = cr_value 
    # e_clean=[(3, 0, 1.0, 0.5),
    #           (0, 2, 1.0, 0.7),
    #           (1, 3, 1.0, 0.3)] #Edges to clean
    influence_set = {}
    ecl = []
    for i,e in enumerate(probGraph.Edges): 
        ecl.append((e[0],e[1],probGraph.weights[e],probGraph.get_prob(e)))
    start_execution_time = time()
    for s,t in queries:
        # e_clean = [(e[0],e[1],e[2]['weight'],e[2]['prob']) for e in G.edges(data=True) if e[2]['prob']<= p_max and e[2]['prob']>=p_min]
        # if args.verbose: print("Candidate edges: ", e_clean)
        index = {}
        # e_clean = deepcopy(ecl)
        e_clean = ecl
        for e in e_clean:
            u,v = e[0],e[1]
            length_v_all = nx.single_source_shortest_path_length(G, v,cutoff=d)
            length_all_u = dict(nx.single_target_shortest_path_length(G, u,cutoff=d))
            for s,t in combinations(nodes,2):
                dv_t = length_v_all[t]
                dv_s = length_v_all[s]
                dt_u = length_all_u[t]
                ds_u = length_all_u[s]
                uv_weight = e[2]
                if (ds_u+uv_weight+dv_t<=d) or (dv_s + uv_weight+dt_u <= d):
                    influence_set[(u,v)] = influence_set.get((u,v),[])
                    influence_set[(u,v)].append((s,t))
        print('length of influence set: ',len(influence_set))
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
        # for v in G.nodes():
        #     try:
        #         ds_v=nx.dijkstra_path_length(G, v, s, weight='weight')
        #     except NetworkXNoPath:
        #         ds_v = 10000000
        #     try:
        #         dt_v=nx.dijkstra_path_length(G, v, t, weight='weight')
        #     except NetworkXNoPath:
        #         dt_v = 10000000
        #     e_copy=e_clean.copy()
        #     if ds_v+dt_v>d:
        #         for a, b, w, p in e_copy:
        #             if a==v or b==v:
        #                 e_clean.remove((a,b,w,p))
                        # print('removed: ',(a,b,w,p))
                        

        # print("Candidate edges after pruning: ", e_clean)
        estar = []
        if args.utype == 'c1': # Non-adaptive
            # estar = find_e_nonadaptive(G, s, t, d, e_clean, budget = budget)
            raise Exception("Non-adaptive query not supported")
        else:
            for i,e in enumerate(e_clean): # re-index
                index[(e[0],e[1])] = i
            r0 = compute_approx_reach(s,t,d,probGraph=probGraph)
            H0 = h(r0) + h(1-r0)
            # print('ro = ',r0)
            for k in range(args.budget):
                print('selecting ',k,'-th edge')
                e=find_e_adaptive(G, s, t, d, influence_set, e_clean, probGraph=probGraph) 
                e = (e[0],e[1])
                estar.append(e)
                # print(e)
                # print('index: ',index)
                # print(index[(e[0],e[1])])
                e_clean.pop(index[e])
                for i,edge in enumerate(e_clean): # re-index
                    index[edge] = i
                # print(cr_dict)
                # print(probGraph.edict)
                probGraph.update_edge_prob(e[0],e[1],cr_dict[e]) # Use crowd knowledge to update p(e*)
                # probGraph.edict[e] = cr_dict[e]
                G.remove_edge(*e)
        r_end = compute_approx_reach(s,t,d,probGraph=probGraph)
        # print('r_end: ',r_end)
        H_end = h(r_end) + h(1-r_end)
        # if args.verbose:
        #     if len(estar==1):
        #         print("Execution completed e*: (",e_star[0][0],", ",e_star[0][1], ")")
        #     else:
        #         for e in e_star:
        #             print("Execution completed e*: (",e[0],", ",e[1], ")")

            # nx.draw(G,with_labels = True)
            # plt.savefig('graph_TKDE.png')
        if args.verbose:
            print('Uncertainty before cleaning: ',compute_uncertainty(probGraph,s,t,d))
            print('Uncertainty after cleaning: ',compute_uncertainty(probGraph,s,t,d,aftercleaning=True,estar=estar))
        
        a = ApproximateAlgorithm(probGraph,Query(probGraph, qtype='reach_d',args = {'u':s,'v':t, 'd':d}))
        a.algostat['execution_time'] = 0
        a.algostat['result'] = {}
        a.algostat['k'] = budget
        a.algostat['algorithm'] = 'TKDE17'
        a.algostat['result']['H0'] = H0
        a.algostat['result']['edges'] = estar
        a.algostat['result']['H*'] = H_end
        a.algostat['execution_time'] = time() - start_execution_time
        a.algostat['DeltaH'] = a.algostat['result']['H0'] - a.algostat['result']['H*']
        a.algostat['|DeltaH|'] = abs(a.algostat['result']['H0'] - a.algostat['result']['H*'])
        del a.algostat['support']
        output = {}
        for k in a.algostat['result']:
            if k == 'edges':
                output[k] = str(a.algostat['result'][k])
            else:
                output[k] = a.algostat['result'][k]
        for k in a.algostat.keys():
            if k!='result':
                output[k] = a.algostat[k]
        csv_name = 'CRureduct/TKDE_' + args.dataset + "_" + str(budget)+"_maxd_"+str(d)+'.csv'
        # csv_name = 'output/measure_' + args.dataset + "_" + args.algo + "_" + args.property + "_" + args.queryf.split("/")[-1].split("_")[-1] + '.csv'
        if os.path.exists(csv_name):
            result_df = pd.read_csv(csv_name)
        else:
            os.system('mkdir -p CRureduct/')
            result_df = pd.DataFrame()
        print(output)
        result = pd.concat([result_df, pd.DataFrame(output,index = [0])])
        print(result)
    # result.to_csv(csv_name, header=True, index=False)
# python AlgoTKDE17.py -d test -s 2 -t 3 -maxd 3 -b 2