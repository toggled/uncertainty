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
parser.add_argument("-a", "--algo", type=str, default="greedy") # exact/greedy/greedymem 
parser.add_argument("-u",'--utype',type = str, default = 'o1')
parser.add_argument("-k",'--k',type = int, default = 1)
parser.add_argument("-v", "--verbose", action='store_true')
parser.add_argument('-s','--source',type = str, default = None)
parser.add_argument('-t','--target',type = str, default = None)
parser.add_argument('-pr','--property',type = str, default = 'tri', help = "either tri/diam/reach")
# parser.add_argument('-va','--variant',type = str, default = 'exact',help = 'Either exact/appr')
parser.add_argument("-K",'--K',type = int, default = 10, help='#of Possible world samples')
parser.add_argument("-ea", "--est_algo", type=str, default="appr") # exact/appr/eappr/mcbfs/pTmcbfs/mcdij/pTmcdij/rss/pTrss/mcapproxtri 
parser.add_argument('-q','--queryf', type=str,help='query file',default = None) # 'data/queries/ER/ER_15_22_2.queries'
parser.add_argument('-b','--bucketing',type = int, help='Whether to compute bucketed entropy or not', default = 0) # only tri query is supported
parser.add_argument("-dh",'--hop',type = int, default = -1) # <d-hop reach
parser.add_argument("-db",'--debug', action = 'store_true')
parser.add_argument('-mq','--maxquery',type = int,help='#query pairs to take, maximum = -1 means All queries',default=-1)
parser.add_argument("-th",'--trackH', action = 'store_true')
# parser.add_argument("-t", "--thread", help="index of thread", default=-1, type=int) 

opt_N_dict = {
    'default': {'reach': 100,'sp': 100},
    'ER_15_22': 
        {'reach': 161, 'sp': 286, 'tri': 466},
    'biomine': {'reach': 171},
    'flickr': {'tri': 76},
    'papers': {'reach': 71},
    'products': {'reach': 46},
    'restaurants': {'reach': 156}
}
opt_T_dict = {
    'default': {'reach': 10,'sp':10},
    'ER_15_22': {'reach': 6, 'sp': 11, 'tri': 11},
    'biomine': {'reach': 10},
    'flickr': {'tri': 51},
    'papers': {'reach': 10},
    'products': {'reach': 4},
    'restaurants': {'reach': 6}
}
# Demo usages:
# Reachability query from x to u in default dataset using sampling: N = 10, T = 10
# python measure_main.py -d default -a appr -pr reach -s x -t u -N 10 -T 10
args = parser.parse_args()
os.environ['precomp'] = ''
os.environ['time_seed'] = 'True' # If we set this, each run will generate a different sequence of possible worlds from the previous run.
dhopreach = [False,True][args.hop>0] 
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
    if args.algo == 'exact': #
        a = Algorithm(G,Query)
        print("Exact algorithm:")
        a.Bruteforce(k = args.k, update_type=args.utype, verbose = args.verbose)
    # elif args.algo == 'greedy+mem(exact)':
    #     a = Algorithm(G,Query, debug = args.debug)
    #     print("Greedy algorithm (w/ exact mem.): ")
        # a.algorithm5(k = args.k, update_type=args.utype, verbose = args.verbose)
    elif args.algo == 'greedy': #
        a = ApproximateAlgorithm(G,Query, debug = args.debug)
        print("Greedy algorithm (w/o mem.): ")
        # a.greedy(k = args.k, update_type=args.utype, verbose = args.verbose)
        a.greedy(property = Query.qtype, algorithm = args.est_algo, k = args.k, \
                     N = opt_T_dict[args.dataset][args.property], T = opt_T_dict[args.dataset][args.property],\
                     update_type=args.utype, verbose = args.verbose, track_H = args.trackH)
    elif args.algo == 'greedymem': #
        a = ApproximateAlgorithm(G,Query, debug = args.debug)
        a.algorithm5(property = Query.qtype, algorithm = args.est_algo, k = args.k, K = args.K, 
                     N = opt_T_dict[args.dataset][args.property], T = opt_T_dict[args.dataset][args.property],\
                    update_type=args.utype, verbose = args.verbose, track_H = args.trackH)
    # elif args.algo == 'greedymem_re':
    #     raise ValueError("do not use this option.")
    #     a = ApproximateAlgorithm(G,Query)
        # a.algorithm6(property = Query.qtype, algorithm = args.est_algo, \
        #              k = args.k, K = args.K, update_type=args.utype, verbose = args.verbose)
    else:
        raise Exception("Invalid algorithm (-a) option.")
    # print(a.algostat)
    # if args.verbose:
    #     G.plot_probabilistic_graph()
    os.system('mkdir -p ureduct/')
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
    output['P'] = Query.qtype
    output['algorithm'] = args.algo
    # output['variant'] = args.variant 
    output['K'] = ["None",args.K][args.algo == 'greedy' or args.algo.startswith('greedymem')]

    for k in a.algostat['result']:
        if k == 'edges':
            output[k] = str(a.algostat['result'][k])
        else:
            output[k] = a.algostat['result'][k]
    for k in a.algostat.keys():
        if k== 'history_deltaH':
            # output[k] = ' '.join([str(i) for i in a.algostat['history_deltaH']])
             output[k] = str(a.algostat['history_deltaH'])
        elif k =='result':
            pass
        else:
            output[k] = a.algostat[k]
    if 'support' in output:
        del output['support']
    print(output)
    if (not args.verbose and not args.debug):
        # csv_name = 'ureduct/res_k'+str(args.k)+'.csv'
        if args.queryf:
            csv_name = 'ureduct/reduce_k_'+str(args.k)+"_K_"+str(args.K)+"_" + args.dataset + "_" + args.algo + "_" + Query.qtype + "_" + args.queryf.split("/")[-1].split("_")[-1] + '.csv'
        else:
            csv_name = 'ureduct/res_k'+str(args.k)+'.csv'
        if os.path.exists(csv_name):
            result_df = pd.read_csv(csv_name)
        else:
            result_df = pd.DataFrame()
        # print(output)
        # print(pd.DataFrame(output,index = [0]).head())
        result = pd.concat([result_df, pd.DataFrame(output,index = [0])])
        result.to_csv(csv_name, header=True, index=False)
        print(result.head(10))
    # if args.debug:
    #     print(output['M'])

# if __name__== '__main__':
#     dataset = 'default'
#     property = 'reach'
#     source = 's'
#     target = 'y'
#     dhopreach = False 
#     hop = -1
#     k = 1 
#     G = get_dataset(dataset)
#     os.environ['time_seed'] = 'True'
#     if property == 'reach':
#         if dhopreach:
#             print('#<d-hop reachability (',source,',',target,')')
#             Q = Query(G, 'reach_d',args = {'u':source,'v':target, 'd':hop})
#         else:
#             print('Reachability(',source,',',target,')')
#             Q = Query(G,'reach',{'u':source,'v':target})

#     if property == 'sp':
#         print('Shortest path')
#         Q = wQuery(G,'sp',{'u':source,'v':target})
#     if property == 'tri':
#         print('#Triangles')
#         Q = Query(G,'tri')
#     a = Algorithm(G, Q, debug = True)
#     print("Greedy algorithm (w/ exact mem.): ")
#     a.algorithm5(k = k, update_type='o1', verbose = False)
#     M = a.algostat['M']
#     hatMList = []
#     for K in range(2,10):
#         Q = Query(G,'reach',{'u':source,'v':target})
#         a = ApproximateAlgorithm(G, Q, debug = True)
#         a.algorithm5(property = Q.qtype, algorithm = 'exact', \
#                         k = k, K = K, update_type='o1', verbose = False)
#         hatMList.append(a.algostat['M'])
#         print(a.algostat['result']['edges'])
#     print(M)
#     print(hatMList[0])
#     print(hatMList[-1])

if __name__== '__main__':
    if args.queryf is None: # single query mode
        G = get_dataset(args.dataset)
        if args.property == 'reach':
            if dhopreach:
                print('#<d-hop reachability (',args.source,',',args.target,')')
                Query = Query(G, 'reach_d',args = {'u':args.source,'v':args.target, 'd':args.hop})
            else:
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
        queries = get_queries(queryfile = args.queryf, maxQ = args.maxquery)
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
                    if dhopreach:
                        print('#<d-hop reach')
                        Query = multiGraphQuery(G, qtype='reach_d',args = {'u':s,'v':t, 'd':args.hop})
                    else: # normal reachability query
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
                if dhopreach:
                    Querylist = [Query(G,args.property,{'u':s,'v':t,'d':args.hop}) for s,t in queries]
                else:     
                    Querylist = [Query(G,args.property,{'u':s,'v':t}) for s,t in queries]
            elif args.property == 'tri':
                Q = Query(G,'tri')
                Q.bucketing = args.bucketing 
                Querylist = [Q]
            else:
                raise Exception("Invalid graph property (-pr) ")  
            for Query in Querylist:
                singleQuery_singleRun(G, Query)

# python reduce_main.py -d test -a greedymem -k 2 -s 2 -t 3 -dh 3 -K 10 -pr reach -a appr
# python reduce_main.py -d test -a greedymem -k 2 -s 2 -t 3 -dh 3 -K 10 -pr reach -ea appr
# python reduce_main.py -d test -a greedymem -k 1 -s 2 -t 3 -dh 3 -K 20 -pr reach -ea mcbfs
# python reduce_main.py -d papers -a greedymem -k 1 -s 2 -t 3 -dh 3 -K 20 -pr reach -ea mcbfs -q data/queries/products/products_2.queries


# python reduce_main.py -k 1 -s y -t u -pr reach -a exact
# python reduce_main.py -k 1 -s y -t u -pr reach -a greedy
# python reduce_main.py -k 1 -s y -t u -K 4 -pr reach -a greedymem -ea exact
# python reduce_main.py -k 1 -s y -t u -pr reach -a greedy+mem(exact)
