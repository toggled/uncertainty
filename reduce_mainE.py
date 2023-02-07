""" Main file to make uncertainty reduction more efficient using SPQR subgraphs. """

import argparse,os,json
# from unittest import result
import networkx as nx
from src.utils import get_dataset,draw_possible_world,save_dict
from src.algorithm import Algorithm,ApproximateAlgorithm
from src.query import Query
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, default="default")
parser.add_argument("-a", "--algo", type=str, default="bruteforce") 
parser.add_argument("-u",'--utype',type = str, default = 'o1')
parser.add_argument("-k",'--k',type = int, default = 1)
parser.add_argument("-v", "--verbose", action='store_true')
parser.add_argument('-s','--source',type = str, default = None)
parser.add_argument('-t','--target',type = str, default = None)
parser.add_argument('-pr','--property',type = str, default = 'tri', help = "either tri/diam/reach")
parser.add_argument('-va','--variant',type = str, default = 'exact',help = 'Either exact/appr')
parser.add_argument("-K",'--K',type = int, default = 100, help='#of Possible world samples')
# parser.add_argument("-t", "--thread", help="index of thread", default=-1, type=int) 

# Demo usages:
# Reachability query from x to u in default dataset using sampling: N = 10, T = 10
# python measure_main.py -d default -a appr -pr reach -s x -t u -N 10 -T 10


args = parser.parse_args()
G = get_dataset(args.dataset)
# G.plot_probabilistic_graph()
# print(G.get_num_edges())
# print(G.get_num_vertices())
# G.plot_possible_world_distr()
# G.plot_probabilistic_graph()
# for g in G.get_Ksample(K = 4):
#     print(g.edges)
#     draw_possible_world(g)

# Query = Query(G,'num_triangles')
# Query = Query(G,'diam')
# Query.eval(None,None,None)
if args.property == 'reach':
    print('Reachability(',args.source,',',args.target,')')
    Query = Query(G,'reach',{'u':args.source,'v':args.target})
if args.property == 'diam':
    print('diameter')
    Query = Query(G,'diam')
if args.property == 'tri':
    print('#Triangles')
    Query = Query(G,'tri')
# Query = Query(G,'reach',{'u':'a','v':'c'})
# Query.eval()
# print(Query.get_distribution())
# print(Query.compute_entropy())
# Query.distr_plot()

if args.variant == 'exact':
    a = Algorithm(G,Query)
    if args.algo == 'bruteforce':
        print("Bruteforce:")
        a.Bruteforce(k = args.k, update_type=args.utype, verbose = args.verbose)
    if args.algo == 'greedyex':
        print("Algorithm3:")
        a.algorithm3(k = args.k, update_type=args.utype, verbose = args.verbose)
    if args.algo == 'greedyct':
        print("Algorithm5:")
        a.algorithm5(k = args.k, update_type=args.utype, verbose = args.verbose)

# elif args.variant == 'exact_adap':
#     pass 

# elif args.variant == 'exact_nonadap':
#     pass 

elif args.variant == 'greedyct_nonadap':
    # select all k edges by running GreedyH-ApprHmem, only after conduct crowdsourcing of selected k edges. 
    a = ApproximateAlgorithm(G,Query)
    if args.algo == 'greedyct':
        print("Algorithm5-S:")
        assert (args.utype=='o2')
        a.algorithm5(k = args.k, K = args.K, update_type=args.utype, verbose = args.verbose)
        E_star = a.algostat['result']['edges']
        for e in E_star:
            G.edge_update(e[0],e[1],type = args.utype)
        Query.reset(G)
        Hstar = Query.compute_entropy()
        print('Reduction (|H0-Hstar|) = ',a.algostat['result']['H0']-Hstar)

    
elif args.variant == 'greedyct_adap':
    # select the top-1 edge by running GreedyH-ApprHmem, conduct crowdsourcing of the selected edge, 
    # update the input graph based on crowdsourcing result, 
    # then again select the top-1 edge by running GreedyH-ApprHmem on the updated graph -- repeat this for k times. 
    pass 
else:
    a = ApproximateAlgorithm(G,Query)
    if args.algo == 'greedyct':
        print("Algorithm5-S:")
        a.algorithm5(k = args.k, K = args.K, update_type=args.utype, verbose = args.verbose)
        
    # print(a.algostat)
# print(a.algostat)
if args.verbose:
    G.plot_probabilistic_graph()
os.system('mkdir -p output/')
# result_fname = 'output/'+args.dataset +'_'+args.algo+"_"+args.utype+'.pkl'
# save_dict(a.algostat, result_fname)
output = {}
output['source'] = str(args.source)
output['target'] = str(args.target)
output['dataset'] = args.dataset
output['P'] = args.property
output['variant'] = args.variant 
output['K'] = ["None",args.K][args.variant == 'appr']

for k in a.algostat['result']:
    if k == 'edges':
        output[k] = str(a.algostat['result'][k])
    else:
        output[k] = a.algostat['result'][k]
for k in a.algostat.keys():
    if k!='result':
        output[k] = a.algostat[k]
print(output)
# print('Expected reduction: ',self.algostat['result']['H0'] - self.algostat['result']['H0']*0.5 + )
if (not args.verbose):
    csv_name = 'output/res_k'+str(args.k)+'.csv'
    if os.path.exists(csv_name):
        result_df = pd.read_csv(csv_name)
    else:
        result_df = pd.DataFrame()
    # print(output)
    # print(pd.DataFrame(output,index = [0]).head())
    result = pd.concat([result_df, pd.DataFrame(output,index = [0])])
    # result.to_csv(csv_name, header=True, index=False)
    print(result.head())