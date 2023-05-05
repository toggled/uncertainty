import argparse,os
import networkx as nx
from src.utils import *
from src.algorithm import Algorithm,ApproximateAlgorithm
from src.query import Query,wQuery,multiGraphQuery,multiGraphwQuery
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, default="ER_15_22")
parser.add_argument("-a", "--algo", type=str, default="appr", help = "exact/appr/eappr") 
parser.add_argument("-N",'--N',type = int, default = 1, help = '#of batches')
parser.add_argument("-T",'--T',type = int, default = 5, help= '#of Possible worlds in a batch')
parser.add_argument("-v", "--verbose", action='store_true')
parser.add_argument('-s','--source',type = str, default = None)
parser.add_argument('-t','--target',type = str, default = None)
parser.add_argument('-q','--queryf', type=str,help='query file',default = 'data/queries/ER/ER_15_22_2.queries')
parser.add_argument('-mq','--maxquery',type = int,help='#query pairs to take, maximum = -1 means All queries',default=-1)
parser.add_argument('-pr','--property',type = str, default = 'sp', help = "either tri/sp/reach")
parser.add_argument('-pre','--precomputed',type = int, help='Use pre-computed possible world support value or not', default = 0)
# Demo usages:
# Reachability query from x to u in default dataset using sampling: N = 10, T = 10
# python measure_main.py -d default -a appr -pr reach -s x -t u -N 10 -T 10

args = parser.parse_args()
# if 'precomp' in os.environ:
#     os.environ['precomp'] = ''
if args.precomputed:
    os.environ['precomp'] = "pre/"+args.dataset+"_"+args.queryf.split("/")[-1].split('.')[0]
    old = os.environ['precomp']
else:
    os.environ['precomp'] = ''
debug = (args.source is not None) and (args.target is not None)

def singleRun(G,Query, save = True):
    if args.algo == 'exact':
        a = Algorithm(G,Query)
        a.measure_uncertainty()
    
    elif args.algo == 'appr':
        a = ApproximateAlgorithm(G,Query)
        a.measure_uncertainty(N=args.N, T = args.T)
    else:
        a = ApproximateAlgorithm(G,Query)
        a.measure_uncertainty(N=args.N, T = args.T)
        a.algostat['algorithm'] = args.algo

    # print(a.algostat)
    if args.verbose:
        G.plot_probabilistic_graph()
    os.system('mkdir -p output/')
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
    G.name = args.dataset

if debug: print(args.property,' (',args.source,',',args.target,')')
if debug:
    Query = Query(G,args.property,{'u':args.source,'v':args.target})
else:
    if args.algo != 'eappr':
        if args.property == 'sp':
            Querylist = [wQuery(G,args.property,{'u':s,'v':t}) for s,t in queries]
        if args.property == 'reach':
            Querylist = [Query(G,args.property,{'u':s,'v':t}) for s,t in queries]

if debug: # Run algorithm for single query (Debugging purposes)
    singleRun(G,Query)
else: # Run algorithms for all the queries
    # print(args.algo)
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
            if args.precomputed:
                os.environ['precomp'] = old
                if args.property != 'tri':
                    os.environ['precomp'] += ("_eappr_"+str(args.property)+"_"+str(Query.u)+"_"+str(Query.v))
                else:
                    os.environ['precomp']+='_eappr_tri'
                os.system('mkdir -p '+os.environ["precomp"])
                print('precomputed support value location: ',os.environ['precomp'])  
            singleRun(G, Query)
            Query.clear()
    else:
        if args.property == 'tri':
            print('#Triangles')
            Querylist = [Query(G,'tri')]
        for Query in Querylist:
            if args.precomputed:
                os.environ['precomp'] = old
                if args.property != 'tri':
                    os.environ['precomp'] += ("_appr_"+str(args.property)+"_"+str(Query.u)+"_"+str(Query.v))
                else:
                    os.environ['precomp']+='_appr_tri'
                os.system('mkdir -p '+os.environ["precomp"])
                print('precomputed support value location: ',os.environ['precomp'])  
            singleRun(G, Query)

            Query.clear()