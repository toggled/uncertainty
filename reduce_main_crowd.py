import argparse,os,json
# from unittest import result
import networkx as nx
from src.utils import *
from src.algorithm import Algorithm,ApproximateAlgorithm
from src.query import Query,wQuery,multiGraphQuery,multiGraphwQuery
import pandas as pd
from datetime import datetime

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, default="default")
parser.add_argument("-a", "--algo", type=str, default="exact") # exact/greedy/greedymem 
parser.add_argument("-u",'--utype',type = str, default = 'o1')
parser.add_argument("-k",'--k',type = int, default = 1)
parser.add_argument("-v", "--verbose", action='store_true')
parser.add_argument('-s','--source',type = str, default = None)
parser.add_argument('-t','--target',type = str, default = None)
parser.add_argument('-pr','--property',type = str, default = 'tri', help = "either tri/diam/reach")
# parser.add_argument('-va','--variant',type = str, default = 'exact',help = 'Either exact/appr')
parser.add_argument("-K",'--K',type = int, default = 10, help='#of Possible world samples')
parser.add_argument("-ea", "--est_algo", type=str, default="exact/appr/eappr/mcbfs/pTmcbfs/mcdij/pTmcdij/rss/pTrss/mcapproxtri") 
parser.add_argument('-q','--queryf', type=str,help='query file',default = None) # 'data/queries/ER/ER_15_22_2.queries'
parser.add_argument('-b','--bucketing',type = int, help='Whether to compute bucketed entropy or not', default = 0) # only tri query is supported

# parser.add_argument("-t", "--thread", help="index of thread", default=-1, type=int) 

# Demo usages:
# Reachability query from x to u in default dataset using sampling: N = 10, T = 10
# python measure_main.py -d default -a appr -pr reach -s x -t u -N 10 -T 10
args = parser.parse_args()
os.environ['precomp'] = ''
os.environ['time_seed'] = 'True'
runProbTree = (args.algo == 'eappr' or args.algo.startswith('pT')) # True if load precomputed ProbTree subgraph
print(args)
# G = get_dataset(args.dataset)
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
# if args.variant == 'exact':
#     a = Algorithm(G,Query)
#     if args.algo == 'bruteforce':
#         print("Bruteforce:")
#         a.Bruteforce(k = args.k, update_type=args.utype, verbose = args.verbose)
#     if args.algo == 'greedyex':
#         print("Algorithm3:")
#         a.algorithm3(k = args.k, update_type=args.utype, verbose = args.verbose)
#     if args.algo == 'greedyct':
#         print("Algorithm5:")
#         a.algorithm5(k = args.k, update_type=args.utype, verbose = args.verbose)
# else:
#     a = ApproximateAlgorithm(G,Query)
#     if args.algo == 'greedyct':
#         print("Algorithm5-S:")
#         a.algorithm5(property = args.property, algorithm = args.est_algo, \
#                      k = args.k, K = args.K, update_type=args.utype, verbose = args.verbose)
#     if args.algo == 'greedyct2':
#         print("Algorithm6-S:")
#         a.algorithm6(k = args.k, K = args.K, update_type=args.utype, verbose = args.verbose)
    # print(a.algostat)
def singleQuery_singleRun(G,Query):
    if args.algo == 'exact':
        a = Algorithm(G,Query)
        print("Exact algorithm:")
        a.Bruteforce(k = args.k, update_type=args.utype, verbose = args.verbose)
    elif args.algo == 'greedy':
        a = Algorithm(G,Query)
        print("Greedy algorithm (w/o mem.): ")
        a.algorithm3(k = args.k, update_type=args.utype, verbose = args.verbose)
    elif args.algo == 'greedymem':
        a = ApproximateAlgorithm(G,Query)
        a.algorithm5(property = args.property, algorithm = args.est_algo, \
                     k = args.k, K = args.K, update_type=args.utype, verbose = args.verbose)
        # a.algorithm6(property = args.property, algorithm = args.est_algo, \
        #              k = args.k, K = args.K, update_type=args.utype, verbose = args.verbose)
    else:
        raise Exception("Invalid algorithm (-a) option.")
    # print(a.algostat)
    # if args.verbose:
    #     G.plot_probabilistic_graph()
    os.system('mkdir -p CRureduct/')
    # result_fname = 'output/'+args.dataset +'_'+args.algo+"_"+args.utype+'.pkl'
    # save_dict(a.algostat, result_fname)
    output = {}
    if args.property == 'tri':
        output['source'] = str(None)
        output['target'] = str(None)
    else:
        output['source'] = str(Query.u)
        output['target'] = str(Query.v)
    output['dataset'] = args.dataset
    output['P'] = args.property
    # output['variant'] = args.variant 
    output['K'] = ["None",args.K][args.algo == 'greedy' or args.algo == 'greedymem']

    for k in a.algostat['result']:
        if k == 'edges':
            output[k] = str(a.algostat['result'][k])
        else:
            output[k] = a.algostat['result'][k]
    for k in a.algostat.keys():
        if k!='result':
            output[k] = a.algostat[k]
    if 'support' in output:
        del output['support']
    print(output)
    if (not args.verbose):
        # csv_name = 'ureduct/res_k'+str(args.k)+'.csv'
        csv_name = 'CRureduct/reduce_k_'+str(args.k)+"_K_"+str(args.K)+"_" + args.dataset + "_" + args.algo + "_" + args.property + "_" + args.queryf.split("/")[-1].split("_")[-1] + '.csv'
        if os.path.exists(csv_name):
            result_df = pd.read_csv(csv_name)
        else:
            result_df = pd.DataFrame()
        # print(output)
        # print(pd.DataFrame(output,index = [0]).head())
        result = pd.concat([result_df, pd.DataFrame(output,index = [0])])
        result.to_csv(csv_name, header=True, index=False)
        print(result.head())

if __name__== '__main__':
    if args.queryf is None: # single query mode
        G = get_dataset(args.dataset)
        if args.property == 'reach':
            print('Reachability(',args.source,',',args.target,')')
            Query = Query(G,'reach',{'u':args.source,'v':args.target})
        if args.property == 'sp':
            print('Shortest path')
            Query = wQuery(G,'sp',{'u':args.source,'v':args.target})
        if args.property == 'tri':
            print('#Triangles')
            Query = Query(G,'tri')
        singleQuery_singleRun(G, Query)
        # Query = Query(G,'reach',{'u':'a','v':'c'})
        # Query.eval()
        # print(Query.get_distribution())
        # print(Query.compute_entropy())
        # Query.distr_plot()
    else: # querySet mode
        assert (args.queryf is not None)
        assert (os.path.isfile(args.queryf))
        queries = get_queries(queryfile = args.queryf) 
        args.queryf
        if runProbTree: # Efficient variant of algorithm 2 requires pre-computed Prob Tree subgraph.
            whichquery = args.queryf.split('.')[0].split('_')[-1]
            rsubgraphpaths = [ 'data/maniu/'+args.dataset+'_'+whichquery+'_subg/'+dataset_to_filename[args.dataset].split('/')[-1]+'_query_subgraph_'+s+'_'+t+'.txt' \
                            for s,t in queries]
            for subpath,q in zip(rsubgraphpaths,queries):
                if (not os.path.isfile(subpath)):
                    raise Exception('representative subgraph: ',subpath,' missing!')
                G = get_decompGraph(args.dataset,None,None,subpath)
                # print('decomp graph: ',G)
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
                    Query.bucketing = args.bucketing 
                # if args.precomputed:
                #     os.environ['precomp'] = old
                #     if args.property == 'sp' or args.property == 'reach':
                #         os.environ['precomp'] += ("_"+args.algo+"_"+str(args.property)+"_"+str(Query.u)+"_"+str(Query.v))
                #     if args.property == 'tri':
                #         os.environ['precomp']+='_'+args.algo+'_tri'
                #     os.system('mkdir -p '+os.environ["precomp"])
                #     print('precomputed support value location: ',os.environ['precomp'])  
                singleQuery_singleRun(G, Query)
                Query.clear()
        else: # Exact and normal variant of algorithm 2 requires original uncertain graph
            G = get_dataset(args.dataset)
            G.name = args.dataset
            if args.property == 'sp':
                Querylist = [wQuery(G,args.property,{'u':s,'v':t}) for s,t in queries]
            elif args.property == 'reach': # For reachability query, we ignore edge weights.  
                Querylist = [Query(G,args.property,{'u':s,'v':t}) for s,t in queries]
            elif args.property == 'tri':
                Q = Query(G,'tri')
                Q.bucketing = args.bucketing 
                Querylist = [Q]
            else:
                raise Exception("Invalid graph property (-pr) ")  
            for Query in Querylist:
                singleQuery_singleRun(G, Query)