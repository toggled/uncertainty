""" Script to run all algorithms for all random graph dataset and update types. """
import os, argparse
import src.utils as utils
parser = argparse.ArgumentParser()
parser.add_argument("-k",'--k',type = int, default = 1)
parser.add_argument('-q','--qtype', type = str, default = 'tri')
parser.add_argument('-va','--variant',type = str, default = 'exact',help = 'Either exact/appr')
parser.add_argument("-K",'--K',type = int, default = 100, help='#of Possible world samples')
args = parser.parse_args()

# dataset_list = ['ER_15_22']
# dataset_list = ['ER_10_15']
# dataset_list = ['ER_15_22']
dataset_list = ['ER_5_7','ER_10_15','ER_15_22']
algorithms = ['bruteforce',"greedyex","greedyct"]
# algorithms = ['greedyct']
utypes = ['o1']
k = args.k
if args.variant == 'appr':
    K = str(args.K)
else:
    K = '1' # If variant is exact => use dummy value 1 for K

cmd_list = []

for d in dataset_list:
    for u in utypes:
        if args.qtype == 'reach':
            for dist in utils.queries[d]:
                with open('data/ER/'+d+'_'+str(dist)+'.queries','r') as f:
                    for line in f.readlines():
                        s,t = line.split()
                        for alg in algorithms:
                            cmd = "python main.py "+"-a " +alg+ ' -va '+args.variant+ ' -K ' + K +" -u "+u+ ' -k '+str(k)+' -d '+d + ' -s '+s + ' -t '+t + ' -pr ' +args.qtype
                            cmd_list.append(cmd)
                            # print(cmd)
                            # os.system(cmd)
                        # print('=========')
        else:
            for alg in algorithms:
                cmd = "python main.py "+"-a " +alg+  ' -va '+args.variant+ ' -K ' + K +" -u "+u+ ' -k '+str(k)+' -d '+d + ' -pr ' +args.qtype
                # print(cmd)
                # os.system(cmd)
                cmd_list.append(cmd)

for c in cmd_list:
    print(c)
    os.system(c)