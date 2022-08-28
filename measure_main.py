import argparse,os
import networkx as nx
from src.utils import get_dataset,draw_possible_world,save_dict
from src.algorithm import Algorithm,ApproximateAlgorithm
from src.query import Query
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, default="default")
parser.add_argument("-a", "--algo", type=str, default="exact", help = "exact/appr") 
parser.add_argument("-N",'--N',type = int, default = 1, help = '#of batches')
parser.add_argument("-T",'--T',type = int, default = 1, help= '#of Possible worlds in a batch')
parser.add_argument("-v", "--verbose", action='store_true')
parser.add_argument('-s','--source',type = str, default = None)
parser.add_argument('-t','--target',type = str, default = None)
parser.add_argument('-pr','--property',type = str, default = 'tri', help = "either tri/diam/reach")
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

if args.algo == 'exact':
    print("measure uncertainty exactly")
    a = Algorithm(G,Query)
    a.measure_uncertainty()
    
else:
    a = ApproximateAlgorithm(G,Query)
    a.measure_uncertainty(N=args.N, T = args.T)

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
output['N'],output['T'] = [("None","None"),(args.N,args.T)][args.algo == 'appr']

for k in a.algostat.keys():
    if k!='result' and k!='k': 
        output[k] = a.algostat[k]
# print(output)
if (not args.verbose):
    csv_name = 'output/measure_'+args.dataset+'.csv'
    if os.path.exists(csv_name):
        result_df = pd.read_csv(csv_name)
    else:
        result_df = pd.DataFrame()
    result = pd.concat([result_df, pd.DataFrame(output,index = [0])])
    result.to_csv(csv_name, header=True, index=False)
    print(result.head())