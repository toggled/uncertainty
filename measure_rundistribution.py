""" Script to run all algorithms for all random graph dataset and update types. """
import os, argparse
import src.utils as utils
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('-q','--qtype', type = str, default = 'tri')
parser.add_argument("--thread", help="index of thread", default = 1, type=int)
args = parser.parse_args()

# dataset_list = ['default','ER_5_7','ER_10_15']
# dataset_list = ['ER_5_7','ER_10_15']
# algorithms = ['appr',"exact"]
# N_list = [1,2,3]
# T_list = [10,20,30]

# dataset_list = ['default']
dataset_list = ['ER_5_7','ER_10_15','twitter_s','flickr','biomine']
# dataset_list = ['twitter_s']
algorithms = ['appr']
N_list = [1]
T_list = [10,20,30,40,50]

def f(cmd_list):
    for cmd in cmd_list:
        os.system(cmd)

if __name__ == '__main__':
    cmd_dict = {}
    for d in dataset_list:
        cmd_list = []
        if args.qtype == 'reach':
            for dist in utils.queries[d]:
                if d.startswith('ER'):
                    query_file = 'data/ER/'+d+'_'+str(dist)+'.queries'
                else:
                    query_file = 'data/'+d+'/'+d+'_'+str(dist)+'.queries'
                with open(query_file,'r') as f:
                    for line in f.readlines():
                        s,t = line.split()
                        for alg in algorithms:
                            cmd = "python measure_main.py "+"-a " +alg +' -d '+d + ' -s '+s + ' -t '+t + ' -pr ' +args.qtype
                            if alg == "appr":
                                for N in N_list:
                                    for T in T_list:
                                        cmd2 = cmd + " -N "+str(N) + " -T "+str(T)
                                        cmd_list.append(cmd2)
                            else:
                                cmd_list.append(cmd)
        else:
            for alg in algorithms:
                cmd = "python measure_main.py "+"-a " +alg +' -d '+d + ' -pr ' +args.qtype
                if alg == "appr":
                    for N in N_list:
                        for T in T_list:
                            cmd2 = cmd + " -N "+str(N) + " -T "+str(T)
                            cmd_list.append(cmd2)
                else:
                    cmd_list.append(cmd)
        cmd_dict[d] = cmd_list 
    print(cmd_list)
    print('#queries: ',len(cmd_list))
    # Generate results sequentially (slow)
    # for c in cmd_list:
    #     os.system(c)

    # Generate results in batches in parallel (fast)
    with Pool(args.thread) as p:
        for i in range(0,len(cmd_dict),args.thread):
            p.map(f,[cmd_dict[j] for j in list(cmd_dict.keys())[i:i+args.thread]] )