from src.graph import UGraph,UMultiGraph
import networkx as nx 
from matplotlib import pyplot as plt
import pickle
from math import log2

# num_nodes = {'ER_15_22': ,\
#              'flickr': ,\
#             'biomine':,\
#             'rome':  }
datasets_wgraph = ['maniu_demow','brain_a1','brain_h1','rome','brno','porto','sanfrancisco','test']
datasets_unwgraph = ['maniu_demo','flickr','biomine', 'ER_15_22','papers','products','restaurants', 'default', 'default2','ER_15_22p','ba','wa']

# decompdataset_to_filename = {
#             "maniu_demo_1_4": "decomp/maniu/demo_1_4.txt",
#             "maniu_demow_1_4": "decomp/maniu/demo_1_4.txt"
# }
dataset_to_filename = {
            "maniu_demo": "data/maniu/demo.txt",
            "maniu_demow": "data/maniu/demow.txt",
            "rome": "data/large/road/Rome.graph",
            "brno": "data/large/road/Brno.graph",
            "porto": "data/large/road/Porto.graph",
            "sfco": "data/large/road/SanFrancisco.graph",
            "biomine" : "data/large/biomine.txt",
            "brain_a1": "data/large/brain/a1_summary_graph.txt",
            "brain_h1": "data/large/brain/h1_summary_graph.txt",
            #-----
            "flickr" : "data/large/Flickr.txt",
            # "flickr_s" : "data/Flickr.txt",
            # "twitter_s" : "data/twitter.txt",
            "default" : "data/test.txt",
            "default2": "data/test2.txt",
            'ba':"data/BA/BA_20_16_seed_1.graph",
            'wa':'WA_10_30_seed_2.graph',
            # 'ER_5_7': 'data/ER/ER_5_7.graph',
            # 'ER_10_15': 'data/ER/ER_10_15.graph',
            'test': 'test_graph_TKDE.txt',
            'ER_15_22': 'data/ER/ER_15_22.graph',
            'ER_15_22p': 'data/ER/ER_15_22_plus.graph',
            'papers': 'data/large/crowd/paper_pair.txt',
            'products': 'data/large/crowd/product_pair.txt',
            'restaurants': 'data/large/crowd/restaurant_pair.txt'
    }
# queries = {
#         "maniu_demo": [1],
#         'default': [1],
#         'ER_5_7': [0,2],
#         'ER_10_15': [0,2],
#         'ER_15_22': [0,2,4],
#         # 'twitter_s': [5,10,15,20],
#         "biomine" : [1,2,3,4,5],
#         "flickr" : [1,2,3,4,5],
#         "brain_a1": [1,2,3,4,5],
#         "brain_h1": [1,2,3,4,5]
# }
def get_queries(queryfile,maxQ = -1):
    queries = []

    with open(queryfile,'r') as f:
        count = 0
        for line in f:
            u,v = line.split()
            queries.append([int(u),int(v)])
            count += 1
            if maxQ!=-1 and count>=maxQ:
                break
    return queries 

def is_weightedGraph(dataset):
    if dataset in datasets_unwgraph:
        return False 
    return True 

def get_decompGraph(dataset, source, target, dataset_path = None):
    """ Load representative subgraph (maniu et al.) from Tree decomposition output """
    name = dataset+'_'+str(source)+'_'+str(target)
    G = UMultiGraph()
    print('ProbTree location: ',dataset_path)
    if dataset in datasets_unwgraph:
        if dataset_path is not None:
            try:
                with open(dataset_path,'r') as f:
                    _id = 1
                    for line in f:
                        u,v,w,p = line.split()
                        # Since dataset is unweighted graph, ignore length l
                        # print('Since dataset is unweighted graph, ignore length l')
                        # G.add_edge(u,v, _id, float(p),float(w),construct_nbr=True)
                        G.add_edge(int(u),int(v), _id, float(p),float(w),construct_nbr=True)
                        _id +=1
            except KeyError as e:
                print("Wrong dataset name provided.")
                raise e
            return G 
        else:
            try:
                with open(dataset_path,'r') as f:
                    _id = 1
                    for line in f:
                        u,v,l,w,p = line.split()
                        # Since dataset is unweighted graph, ignore length l
                        # G.add_edge(u,v, _id, float(p),float(w),construct_nbr=True)
                        G.add_edge(int(u),int(v), _id, float(p),float(w),construct_nbr=True)
                        _id +=1
            except KeyError as e:
                print("Wrong dataset name provided.")
                raise e
            return G 

    else:
        try:
            with open(dataset_path,'r') as f:
                _id = 1
                for line in f:
                    u,v,l,w,p = line.split()
                    # Since dataset is weighted graph, length (l) column already contains weight of original edges + pseudo edges.
                    # G.add_edge(u,v, _id, float(p),float(l),construct_nbr=True)
                    G.add_edge(int(u),int(v), _id, float(p),float(l),construct_nbr=True)
                    _id +=1
        except KeyError as e:
            print("Wrong dataset name provided.")
            raise e
        return G 
# @profile
def get_dataset(dataset):
    """
    Load the graph datasets into memory as UGraph(). // The dataset is assumed to be simple Graph.
    """
    has_weight = is_weightedGraph(dataset)
    print('weighted graph? => ',has_weight)
    G = UGraph()
    try:
        with open(dataset_to_filename[dataset]) as f:
            i = 0
            for line in f:
                if dataset.startswith('brain') and i==0: # Ignore first line of these datasets.
                    i+=1
                    continue 
                if not has_weight:
                    u,v,p = line.split()
                    # G.add_edge(u,v,float(p),construct_nbr=True) # For T_loop/N_Loop we memorize nbrs, so set it to false
                    G.add_edge(int(u),int(v),float(p),construct_nbr=True,weight=None)
                else:
                    u,v,w,p = line.split()
                    # G.add_edge(u,v,float(p),weight=float(w),construct_nbr=True) # For T_loop/N_Loop we memorize nbrs, so set it to false
                    G.add_edge(int(u),int(v),float(p),weight=float(w),construct_nbr=True) # For T_loop/N_Loop we memorize nbrs, so set it to false
    except KeyError as e:
        print("Wrong dataset name provided.")
        raise e
    return G 

def draw_possible_world(nx_graph, seed = 1):
    """ Plot a networkx weighted graph whose weight is probability. """
    fig,ax = plt.subplots()
    pos = nx.spring_layout(nx_graph, seed = seed, weight = None)

    nx.draw_networkx_nodes(nx_graph, pos = pos, ax = ax)

    nx.draw_networkx_edges(nx_graph, pos = pos, edgelist = nx_graph.edges, ax = ax)

    labels = nx.get_edge_attributes(nx_graph,'weight')
    nx.draw_networkx_labels(nx_graph, pos = pos, ax = ax, font_size = 14)
    nx.draw_networkx_edge_labels(nx_graph,pos, ax = ax, edge_labels=labels, font_size = 14)
    # plt.title(title)
    fig.tight_layout()
    # if savedrawing:
    #     plt.savefig(filename, bbox_inches = 'tight')
    plt.show()


def save_dict(dict,fname = 'temp/temp.pkl'):
    """ Save a dictionary """
    print('Saving dictionary to: ',fname)
    with open(fname, 'wb') as handle:
        pickle.dump(dict, handle, protocol= 4) 

def load_dict(fname = 'temp/temp.pkl'):
    """ Load a dictionary """
    print('Loading dictionary from: ',fname)
    with open(fname, 'rb') as handle:
        dict = pickle.load(handle)
        return dict 

def save_ugraph(ugraph, fname='temp/graph.pkl'):
    save_dict(ugraph,fname)

def load_ugraph(fname = 'temp/graph.pkl'):
    return load_dict(fname)

ZERO = 10**(-13)
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

# 0 1 0.6
# 0 10 0.7
# 1 10 0.4
# 10 11 0.5
