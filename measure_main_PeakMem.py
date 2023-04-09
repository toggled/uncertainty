import argparse,os
import networkx as nx
from src.utils import *
# from src.utils import get_dataset,get_decompGraph,draw_possible_world,save_dict,get_queries
from src.algorithm import Algorithm,ApproximateAlgorithm
from src.query import Query,wQuery,multiGraphQuery,multiGraphwQuery
import pandas as pd
import tracemalloc
# from memory_profiler import memory_usage


parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, default="ER_15_22")
parser.add_argument("-a", "--algo", type=str, default="appr", help = "exact/appr/eappr") 
parser.add_argument("-N",'--N',type = int, default = 1, help = '#of batches')
parser.add_argument("-T",'--T',type = int, default = 1000, help= '#of Possible worlds in a batch')
parser.add_argument("-v", "--verbose", action='store_true')
parser.add_argument('-s','--source',type = str, default = None)
parser.add_argument('-t','--target',type = str, default = None)
parser.add_argument('-q','--queryf', type=str,help='query file',default = 'data/queries/ER/ER_15_22_2.queries')
parser.add_argument('-mq','--maxquery',type = int,help='#query pairs to take, maximum = -1 means All queries',default=-1)
parser.add_argument('-pr','--property',type = str, default = 'sp', help = "either tri/sp/reach")

# Demo usages:
# Reachability query from x to u in default dataset using sampling: N = 10, T = 10
# python measure_main.py -d default -a appr -pr reach -s x -t u -N 10 -T 10


args = parser.parse_args()
debug = (args.source is not None) and (args.target is not None)

# @profile
def singleRun(G,Query, save = True):
    if args.algo == 'exact':
        print("measure uncertainty exactly")
        tracemalloc.start()
        a = Algorithm(G,Query)
        a.measure_uncertainty()
        current_mem_appr, peak_mem_appr = tracemalloc.get_traced_memory()
        # tracemalloc.reset_peak()
        tracemalloc.stop()
        a.algostat['peak_memB'] = peak_mem_appr/(10**6) # peakMem in MB
    
    elif args.algo == 'appr':
        # tracemalloc.reset_peak()
        
        a = ApproximateAlgorithm(G,Query)
        tracemalloc.start()
        # tracemalloc.clear_traces()
        # tracemalloc.reset_peak()
        a.measure_uncertainty(N=args.N, T = args.T)
        current_mem_appr, peak_mem_appr = tracemalloc.get_traced_memory()
        # tracemalloc.reset_peak()
        # tracemalloc.clear_traces()
        tracemalloc.stop()
        a.algostat['peak_memB'] = peak_mem_appr/(10**6) # peakMem in MB
        # mem = memory_usage((a.measure_uncertainty,(args.N, args.T,)),\
        #                    timestamps=False, interval=0.01,max_usage = True,\
        #                     backend="psutil")
        # a.algostat['peak_memB'] = mem
        # print('max mem: ',mem)
    else:
        a = ApproximateAlgorithm(G,Query)
        tracemalloc.start()
        a.measure_uncertainty(N=args.N, T = args.T)
        current_mem_appr, peak_mem_appr = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        a.algostat['peak_memB'] = peak_mem_appr/(10**6) # peakMem in MB
        a.algostat['algorithm'] = args.algo

    # print(a.algostat)
    if args.verbose:
        G.plot_probabilistic_graph()
    os.system('mkdir -p output/')
    # result_fname = 'output/'+args.dataset +'_'+args.algo+"_"+args.utype+'.pkl'
    # save_dict(a.algostat, result_fname)
    output = {}
    if args.pr == 'tri':
        output['source'] = None
        output['target'] = None
    else:
        output['source'] = str(Query.u)
        output['target'] = str(Query.v)
    output['dataset'] = args.dataset
    output['P'] = args.property
    output['N'],output['T'] = [("None","None"),(args.N,args.T)][args.algo.endswith('appr')]

    for k in a.algostat.keys():
        if k!='result' and k!='k': 
            output[k] = a.algostat[k]
    # print(output)
    if (not args.verbose):
        # csv_name = 'output/measure_'+args.dataset+'.csv'
        csv_name = 'output/measure_' + args.dataset +"_"+ args.algo +"_"+ args.property + '.csv'
        if os.path.exists(csv_name):
            result_df = pd.read_csv(csv_name)
        else:
            result_df = pd.DataFrame()
        result = pd.concat([result_df, pd.DataFrame(output,index = [0])])
        if save: # Save the algorithm run statistics
            result.to_csv(csv_name, header=True, index=False)
            print(result.head())
        else:
            print(result.head())
    a = None 
# Get list of queries
if not debug:
    queries = get_queries(queryfile = args.queryf, maxQ = args.maxquery) 

# Depending on the algorithm to run get the uncertain graph
if args.algo == 'eappr': # Efficient variant of algorithm 2 requires pre-computed representative subgraphs
    whichquery = args.queryf.split('.')[0].split('_')[-1]
    rsubgraphpaths = [ 'data/maniu/'+args.dataset+'_'+whichquery+'_subg/'+dataset_to_filename[args.dataset].split('/')[-1]+'_query_subgraph_'+s+'_'+t+'.txt' \
                     for s,t in queries]
else: # Exact and normal variant of algorithm 2 requires original uncertain graph
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

if debug: print(args.property,' (',args.source,',',args.target,')')
if debug:
    Query = Query(G,args.property,{'u':args.source,'v':args.target})
else:
    if args.algo != 'eappr':
        Querylist = [Query(G,args.property,{'u':s,'v':t}) for s,t in queries]

if args.property == 'tri':
    print('#Triangles')
    Query = Query(G,'tri')
# Query = Query(G,'reach',{'u':'a','v':'c'})
# Query.eval()
# print(Query.get_distribution())
# print(Query.compute_entropy())
# Query.distr_plot()
if debug: # Run algorithm for single query (Debugging purposes)
    singleRun(G,Query)
else: # Run algorithms for all the queries
    if args.algo == 'eappr':
        for subpath,q in zip(rsubgraphpaths,queries):
            if (not os.path.isfile(subpath)):
                raise Exception('representative subgraph: ',subpath,' missing!')
            G = get_decompGraph(args.dataset,None,None,subpath)
            s,t = q
            if args.property == 'reach':
                Query = multiGraphQuery(G,'reach',{'u':s,'v':t})
            if args.property == 'sp':
                if is_weightedGraph(args.dataset):
                    Query = multiGraphwQuery(G,'sp',{'u':s,'v':t}) # only SP requires weighted multigraph query
                else:
                    Query = multiGraphQuery(G,'sp',{'u':s,'v':t})

            if args.property == 'tri':
                Query = multiGraphQuery(G,'tri')
            singleRun(G, Query)
            # cur_mem_usage = memory_usage(-1, interval=0.01, timeout=1)[-1]
            # mem = memory_usage((singleRun,(G,Query,)),\
            #                timestamps=False, interval=0.001,max_usage = True,\
            #                 backend="psutil")
            # print('peakMem: ',mem-cur_mem_usage)
            # Query.clear()
    else:
        print(args.algo)
        for Query in Querylist:
            singleRun(G, Query)
            Query.clear()
            # Query.reset()
        # singleRun(G, Querylist[0])
        # singleRun(G, Querylist[1])
        # singleRun(G, Querylist[2])
        # singleRun(G, Querylist[3])
        # singleRun(G, Querylist[4])

# python measure_main.py -a exact -pr sp
# python measure_main.py -a exact -pr reach
# python measure_main.py -a appr -pr sp
# python measure_main.py -a appr -pr reach
# python measure_main.py -a eappr -pr sp
# python measure_main.py -a eappr -pr reach

