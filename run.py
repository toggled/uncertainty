""" Script to run all algorithms for default dataset and update types. """
import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("-k",'--k',type = int, default = 1)
parser.add_argument('-s','--source',type = str, default = None,help = 'provide source vertex: -s')
parser.add_argument('-t','--target',type = str, default = None, help = 'provide target vertex: -t')
parser.add_argument('-q','--qtype', type = str, default = 'tri', help = "either tri/diam/reach")
parser.add_argument('-va','--variant',type = str, default = 'exact',help = 'Either exact/appr')
parser.add_argument("-K",'--K',type = int, default = 100, help='#of Possible world samples')
args = parser.parse_args()

dataset_list = ['default']
# algorithms = ['bruteforce',"greedyex","greedyct"]
algorithms = ["greedyct"]
utypes = ['o1']
k = args.k

if args.variant == 'appr':
    K = str(args.K)
else:
    K = '1'

if args.qtype == 'reach':
    assert args.source is not None and args.target is not None 

cmd_list = []
for alg in algorithms:
    for d in dataset_list:
        for u in utypes:
            if args.qtype == 'reach':
                cmd = "python main.py "+"-a " +alg + ' -va '+args.variant+ ' -K ' + K + " -u "+u+ ' -k '+str(k)+' -d '+d+' -s '+args.source + ' -t '+args.target + ' -pr ' +args.qtype
                cmd_list.append(cmd)
            else:
                cmd = "python main.py "+"-a " +alg + ' -va '+args.variant+ ' -K ' + K + " -u "+u+ ' -k '+str(k)+' -d '+d + ' -pr ' +args.qtype
                cmd_list.append(cmd)


for c in cmd_list:
    print(c)
    os.system(c)