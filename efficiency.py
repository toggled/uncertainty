import argparse,os,sys
import networkx as nx
from src.utils import get_dataset,get_decompGraph, draw_possible_world,save_dict,is_weightedGraph
from src.algorithm import Algorithm,ApproximateAlgorithm
from src.query import Query,wQuery,multiGraphQuery,multiGraphwQuery
import pandas as pd
import itertools 
parser = argparse.ArgumentParser()

parser.add_argument("-a", "--algo", type=str, default="exact", help = "exact/appr/eappr") 
parser.add_argument("-N",'--N',type = int, default = 1, help = '#of batches')
parser.add_argument("-T",'--T',type = int, default = 1, help= '#of Possible worlds in a batch')
parser.add_argument("-v", "--verbose", action='store_true')
parser.add_argument('-pr','--property',type = str, default = 'tri', help = "either tri/diam/reach")

datasets = ['Flickr','biomine']
query_sets = [str(x) for x in [1,2]]
subgraph_folders = ["_".join(x) for x in itertools.product(datasets,query_sets)]
subgraph_folders = [x+'_subg' for x in subgraph_folders]

# Demo usages:
# Reachability query from x to u in default dataset using sampling: N = 10, T = 10
# python measure_main.py -d default -a appr -pr reach -s x -t u -N 10 -T 10

folder = subgraph_folders[0]
file = os.listdir(folder)[0]
u,v = file.split('.')[0].split('_')[-2:]
u,v = int(u),int(v)
print(folder,' -- ',file)
args = parser.parse_args()

G = get_dataset(folder.lower())

# G.plot_probabilistic_graph()
# print(G.get_num_edges())
# print(G.get_num_vertices())
# G.plot_possible_world_distr()
# G.plot_probabilistic_graph()
# for g in G.get_Ksample(K = 4):
#     print(g.edges)
#     draw_possible_world(g)

# Q = Query(G,'num_triangles')
# Q = Query(G,'diam')
# Q.eval(None,None,None)
if args.property == 'reach':
    print('Reachability(',u,',',v,')')
    if is_weightedGraph(args.dataset):
        Q = wQuery(G,'reach',{'u':u,'v':v})
    else:
        Q = Query(G,'reach',{'u':u,'v':v})

if args.property == 'sp':
    print('SP(',u,',',v,')')
    if is_weightedGraph(args.dataset):
        Q = wQuery(G,'sp',{'u':u,'v':v})
    else:
        Q = Query(G,'sp',{'u':u,'v':v})

if args.property == 'diam':
    sys.exit(1)
    # print('diameter')
    # Q = Query(G,'diam')
if args.property == 'tri':
    print('#Triangles')
    Q = Query(G,'tri')
# Q = Query(G,'reach',{'u':'a','v':'c'})
# Q.eval()
# print(Q.get_distribution())
# print(Q.compute_entropy())
# Q.distr_plot()

if args.algo == 'exact':
    print("measure uncertainty exactly")
    a = Algorithm(G,Query)
    a.measure_uncertainty()
elif args.algo == 'eappr': # efficient approximate measurement of uncertainty
    print("Efficiently measure uncertainty approximately")
    G = UMultiGraph()

    if dataset in datasets_unwgraph:
        try:
            with open(decompdataset_to_filename[name],'r') as f:
                _id = 1
                for line in f:
                    u,v,l,w,p = line.split()
                    # Since dataset is unweighted graph, ignore length l
                    G.add_edge(u,v, _id, float(p),float(w))
                    _id +=1
        except KeyError as e:
            print("Wrong dataset name provided.")
            raise e
        return G 

    else:
        try:
            with open(decompdataset_to_filename[name],'r') as f:
                _id = 1
                for line in f:
                    u,v,l,w,p = line.split()
                    # Since dataset is weighted graph, length (l) column already contains weight of original edges + pseudo edges.
                    G.add_edge(u,v, _id, float(p),float(l))
                    _id +=1
        except KeyError as e:
            print("Wrong dataset name provided.")
            raise e


    # decomp_g = get_decompGraph(args.dataset, args.source, args.target)
    # print(decomp_g.Edges)
    if args.property == 'reach':
        Q = multiGraphQuery(decomp_g,'reach',{'u':args.source,'v':args.target})

    if args.property == 'sp':
        if is_weightedGraph(args.dataset):
            Q = multiGraphwQuery(decomp_g,'sp',{'u':args.source,'v':args.target}) # only SP requires weighted multigraph query
        else:
            Q = multiGraphQuery(decomp_g,'sp',{'u':args.source,'v':args.target})

    if args.property == 'tri':
        Q = multiGraphQuery(decomp_g,'tri')

   
    a = ApproximateAlgorithm(decomp_g,Q)
    a.measure_uncertainty(N=args.N, T = args.T)
    a.algostat['algorithm'] = args.algo
else:
    a = ApproximateAlgorithm(G,Q)
    a.measure_uncertainty(N=args.N, T = args.T)

# print(a.algostat)
if args.verbose:
    G.plot_probabilistic_graph()
os.system('mkdir -p output/')
# result_fname = 'output/'+args.dataset +'_'+args.algo+"_"+args.utype+'.pkl'
# save_dict(a.algostat, result_fname)


output = {}
output['source'] = str(u)
output['target'] = str(v)
output['dataset'] = args.dataset
output['P'] = args.property
output['N'],output['T'] = [("None","None"),(args.N,args.T)][args.algo == 'appr' or args.algo == 'eappr']

for k in a.algostat.keys():
    if k!='result' and k!='k': 
        output[k] = a.algostat[k]
print(output)
if (not args.verbose):
    csv_name = 'output/measure_'+args.dataset+'.csv'
    if os.path.exists(csv_name):
        result_df = pd.read_csv(csv_name)
    else:
        result_df = pd.DataFrame()
    result = pd.concat([result_df, pd.DataFrame(output,index = [0])])
    result.to_csv(csv_name, header=True, index=False)
    # print(result.head())

