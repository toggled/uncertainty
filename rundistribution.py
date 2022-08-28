""" Script to run all algorithms for all random graph dataset and update types. """
import os, argparse
import src.utils as utils
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('-q','--qtype', type = str, default = 'tri')
parser.add_argument("--thread", help="index of thread", default = -1, type=int)
args = parser.parse_args()

# dataset_list = ['default','ER_5_7','ER_10_15','ER_15_22']
# dataset_list = ['default']
dataset_list = ['flickr','biomine']
algorithms = ['bruteforce',"greedyex","greedyct"]
k_list = range(2,10,2)
K_list = [10,20,30,40,50,60]
utypes = ['o1']

def f(cmd_list):
    for c in cmd_list:
        os.system(c)

if __name__ == '__main__':
    cmd_list_all = []
    for k in k_list:
        cmd_list = []
        for u in utypes:
            for d in dataset_list:
                if args.qtype == 'reach':
                    for dist in utils.queries[d]:
                        with open('data/ER/'+d+'_'+str(dist)+'.queries','r') as f:
                            for line in f.readlines():
                                s,t = line.split()
                                for alg in algorithms:
                                    cmd = "python main.py "+"-a " +alg +" -u "+u+ ' -k '+str(k)+' -d '+d + ' -s '+s + ' -t '+t + ' -pr ' +args.qtype
                                    cmd_list.append(cmd)
                                    if alg == "greedyct":
                                        for K in K_list:
                                            cmd2 = cmd + " -va appr -K "+str(K) 
                                            cmd_list.append(cmd2)
                else:
                    for alg in algorithms:
                        cmd = "python main.py "+"-a " +alg+" -u "+u+ ' -k '+str(k)+' -d '+d + ' -pr ' +args.qtype
                        cmd_list.append(cmd)
                        if alg == "greedyct":
                            for K in K_list:
                                cmd2 = cmd + " -va appr -K "+str(K) 
                                cmd_list.append(cmd2)

        cmd_list_all.append(cmd_list)
        # for c in cmd_list:
        #     os.system(c)
    # print(cmd_list_all)
    with Pool(args.thread) as p:
        p.map(f, cmd_list_all)