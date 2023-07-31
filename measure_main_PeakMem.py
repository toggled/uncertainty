import argparse,os
import networkx as nx
from src.utils import *
# from src.utils import get_dataset,get_decompGraph,draw_possible_world,save_dict,get_queries
from src.algorithm import Algorithm,ApproximateAlgorithm
from src.query import Query,wQuery,multiGraphQuery,multiGraphwQuery
import pandas as pd
import tracemalloc
# from memory_profiler import memory_usage
import gc

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, default="ER_15_22")
parser.add_argument("-a", "--algo", type=str, default="appr",help = "exact/appr/eappr/mcbfs/pTmcbfs/mcdij/pTmcdij/rss/pTrss/mcapproxtri")
parser.add_argument("-N",'--N',type = int, default = 1, help = '#of batches')
parser.add_argument("-T",'--T',type = int, default = 5, help= '#of Possible worlds in a batch')
parser.add_argument("-v", "--verbose", action='store_true')
parser.add_argument('-s','--source',type = str, default = None)
parser.add_argument('-t','--target',type = str, default = None)
parser.add_argument('-q','--queryf', type=str,help='query file',default = 'data/queries/ER/ER_15_22_2.queries')
parser.add_argument('-mq','--maxquery',type = int,help='#query pairs to take, maximum = -1 means All queries',default=-1)
parser.add_argument('-pr','--property',type = str, default = 'sp', help = "either tri/sp/reach")
parser.add_argument('-S','--stat',action='store_true')
# Demo usages:
# Reachability query from x to u in default dataset using sampling: N = 10, T = 10
# python measure_main.py -d default -a appr -pr reach -s x -t u -N 10 -T 10


args = parser.parse_args()
print(args)
debug = (args.source is not None) and (args.target is not None)
runProbTree = (args.algo == 'eappr' or args.algo.startswith('pT')) 
os.environ['precomp'] = ''
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
    
    elif (args.algo == 'appr' or args.algo=='eappr'):
        # tracemalloc.reset_peak()
        a = ApproximateAlgorithm(G,Query)
        # gc.collect()
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
        # a.algostat['algorithm'] = 'MC'
        a.algostat['algorithm'] = ['MC','PT-MC'][args.algo=='eappr']
        # print(args.algo)

    elif (args.algo == 'mcbfs' or args.algo == 'pTmcbfs'):
        a = ApproximateAlgorithm(G,Query)
        assert (args.property == 'reach')
        tracemalloc.start()
        a.measure_uncertainty_bfs(N=args.N, T = args.T)
        current_mem_appr, peak_mem_appr = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        a.algostat['peak_memB'] = peak_mem_appr/(10**6)
        a.algostat['algorithm'] = ['MC+BFS','PT-MC+BFS'][args.algo=='pTmcbfs']

    elif (args.algo == 'mcdij' or args.algo == 'pTmcdij'):
        a = ApproximateAlgorithm(G,Query)
        assert (args.property == 'sp')
        tracemalloc.start()
        a.measure_uncertainty_dij(N=args.N, T = args.T)
        current_mem_appr, peak_mem_appr = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        a.algostat['peak_memB'] = peak_mem_appr/(10**6)
        a.algostat['algorithm'] = ['MC+DIJ','PT-MC+DIJ'][args.algo=='pTmcdij']
    
    elif (args.algo == 'mcapproxtri'):
        a = ApproximateAlgorithm(G,Query)
        assert (args.property == 'tri')
        # n = num_nodes()
        # nu = 100 # 1000 # prob of having good estimate is at least 99%
        # eps = 1/sqrt(n) # +-sqrt(n) error will be incurred during tri counting , but with prob at most 1 - ((nu -1)/nu)
        # k = ceil(ln(2*nu)/(2*eps**2))
        # print('approximate triangle counting: nu = ',nu,' eps = ',eps,' n = ',n, ' k = ',k)
        tracemalloc.start()
        a.measure_uncertainty_mctri(N=args.N, T = args.T)
        current_mem_appr, peak_mem_appr = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        a.algostat['peak_memB'] = peak_mem_appr/(10**6)
        a.algostat['algorithm'] = args.algo

    elif (args.algo == 'rss' or args.algo == 'pTrss'):
        a = ApproximateAlgorithm(G,Query)
        assert (args.property == 'reach')
        tracemalloc.start()
        a.measure_uncertainty_rss(N=args.N, T = args.T)
        current_mem_appr, peak_mem_appr = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        a.algostat['peak_memB'] = peak_mem_appr/(10**6)
        a.algostat['algorithm'] = ['RSS','PT-RSS'][args.algo=='pTrss']
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
    if args.property == 'tri':
        output['source'] = None
        output['target'] = None
    else:
        output['source'] = str(Query.u)
        output['target'] = str(Query.v)
    output['dataset'] = args.dataset
    output['P'] = args.property
    output['N'],output['T'] = args.N,args.T

    for k in a.algostat.keys():
        if k!='result' and k!='k': 
            output[k] = a.algostat[k]
    # print(output)
    if (not args.verbose):
        # csv_name = 'output/measure_'+args.dataset+'.csv'
        if args.stat:
            csv_name = 'output/stats_'+args.dataset+'.csv'
        else:
            csv_name = 'output/PeakMem_measure_' + args.dataset + "_" + args.algo + "_" + args.property + "_" + args.queryf.split("/")[-1].split("_")[-1] + '.csv'
        if os.path.exists(csv_name):
            result_df = pd.read_csv(csv_name)
        else:
            result_df = pd.DataFrame()
        result = pd.concat([result_df, pd.DataFrame(output,index = [0])])
        if save: # Save the algorithm run statistics
            result.to_csv(csv_name, header=True, index=False)
            print(result.head(10))
        else:
            print(result.head(10))
        return result
    a = None 
    return None 
# Get list of queries
if not debug:
    queries = get_queries(queryfile = args.queryf, maxQ = args.maxquery) 

# Depending on the algorithm to run get the uncertain graph
if runProbTree: # Efficient variant of algorithm 2 requires pre-computed representative subgraphs
    whichquery = args.queryf.split('.')[0].split('_')[-1]
    rsubgraphpaths = [ 'data/maniu/'+args.dataset+'_'+whichquery+'_subg/'+dataset_to_filename[args.dataset].split('/')[-1]+'_query_subgraph_'+s+'_'+t+'.txt' \
                     for s,t in queries]
else: # Exact and normal variant of algorithm 2 requires original uncertain graph
    G = get_dataset(args.dataset)
    G.name = args.dataset
    if (args.algo == 'appr' or args.algo=='eappr'):
        G.nbrs = None 
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
    if not runProbTree:
        if args.property == 'sp':
            if is_weightedGraph(args.dataset):
                Querylist = [wQuery(G,args.property,{'u':s,'v':t}) for s,t in queries]
            else:
                Querylist = [Query(G,args.property,{'u':s,'v':t}) for s,t in queries]
        if args.property == 'reach':
            Querylist = [Query(G,args.property,{'u':s,'v':t}) for s,t in queries]

# Query = Query(G,'reach',{'u':'a','v':'c'})
# Query.eval()
# print(Query.get_distribution())
# print(Query.compute_entropy())
# Query.distr_plot()
if debug: # Run algorithm for single query (Debugging purposes)
    singleRun(G,Query)
else: # Run algorithms for all the queries
    if runProbTree:
        for subpath,q in zip(rsubgraphpaths,queries):
            if (not os.path.isfile(subpath)):
                raise Exception('representative subgraph: ',subpath,' missing!')
            G = get_decompGraph(args.dataset,None,None,subpath)
            if (args.algo == 'appr' or args.algo=='eappr'):
                # tracemalloc.reset_peak()
                G.nbrs = None 
            s,t = q
            if args.property == 'reach':
                G = G.simplify()
                print(type(G),' ',len(G.nbrs))
                # print('decomp graph: ',type(G),' |ProbTree V| = ',G.nx_format.number_of_nodes(),' |ProbTree E|=',G.nx_format.number_of_edges())
                # Query = multiGraphQuery(G,'reach',{'u':s,'v':t})
            if args.property == 'reach':
                if isinstance(G, UMultiGraph):
                    Q = multiGraphQuery(G,'reach',{'u':s,'v':t})
                else: 
                    Q = Query(G,'reach',{'u':s,'v':t})
            if args.property == 'sp':
                if is_weightedGraph(args.dataset):
                    Q = multiGraphwQuery(G,'sp',{'u':s,'v':t}) # only SP requires weighted multigraph query
                else:
                    Q = multiGraphQuery(G,'sp',{'u':s,'v':t})

            if args.property == 'tri':
                Q = multiGraphQuery(G,'tri')
            result_df = singleRun(G, Q)
            # cur_mem_usage = memory_usage(-1, interval=0.01, timeout=1)[-1]
            # mem = memory_usage((singleRun,(G,Query,)),\
            #                timestamps=False, interval=0.001,max_usage = True,\
            #                 backend="psutil")
            # print('peakMem: ',mem-cur_mem_usage)
            Q.clear()
    else:
        if args.property == 'tri':
            print('#Triangles')
            Q = [Query(G,'tri')]
            Querylist = [Q]
        if args.stat: result_df = None 
        for Query in Querylist:
            result_df = singleRun(G, Query)
            Query.clear()
    if args.stat:
        grp = result_df.groupby(['dataset','P','algorithm'])
        print(grp[['peak_memB','execution_time']].mean())

# python measure_main.py -a exact -pr sp
# python measure_main.py -a exact -pr reach
# python measure_main.py -a appr -pr sp
# python measure_main.py -a appr -pr reach
# python measure_main.py -a eappr -pr sp
# python measure_main.py -a eappr -pr reach

