import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--token", help="token of dir", default="", type=str)
# parser.add_argument("--mpi", action='store_true')
parser.add_argument("--all", action='store_true')
args = parser.parse_args()


os.system("./clean.sh")

path = "NNdhUiT@biggraph.scse.ntu.edu.sg:/data1/Naheed/uncertainty/"

if(args.all):
    os.system("tar -czvf file_to_send.tar.gz uncertainty/* src/* RelComp/* data/* decomp/* *.py *sh *md")
else:
    os.system("tar -czvf file_to_send.tar.gz uncertainty/* src/* RelComp/* *.py *sh *md")
os.system("rsync -vaP file_to_send.tar.gz "+path)


os.system("rm file_to_send.tar.gz")