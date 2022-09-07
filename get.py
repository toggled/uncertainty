import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--token", help="token of dir", default="", type=str)
# parser.add_argument("--mpi", action='store_true')
parser.add_argument("--large", action='store_true')
args = parser.parse_args()


path = "NNdhUiT@biggraph.scse.ntu.edu.sg:/data1/Naheed/uncertainty/output/"
path2 = "NNdhUiT@biggraph.scse.ntu.edu.sg:/data1/Naheed/uncertainty/decomp/"

os.system("rm -r server_output | mkdir -p server_output")
if(os.path.isdir("server_output")):
    os.system("rm server_output/")
os.system("mkdir -p server_output/" )
if(not args.large):
    os.system("rsync -vaP "+path+"/output.tar.gz server_output" +args.token+"/")
    os.system("tar -xvf server_output" +args.token+"/output.tar.gz -C server_output" +args.token+"/")
    os.system("rsync -vaP "+path2+"/decomp.tar.gz server_output" +args.token+"/")
    os.system("tar -xvf server_output" +args.token+"/decomp.tar.gz -C server_output" +args.token+"/")

if(args.large):
    os.system("rsync -vaP "+path+"/*.csv server_output" +args.token+"/")
    
    
    # os.system("mv server_output/mpi" +args.token+"/output/* server_output/mpi" +args.token+"/")
    # pass

# else:
#     path="nscc:/home/projects/11000744/bishwa/xRNN" +args.token+"/" 
#     os.system("mkdir server_output/xRNN"+args.token+"/")
#     os.system("rsync -vaP "+path+"output/output.tar.gz server_output/xRNN"+args.token+"/")
#     os.system("tar -xvf server_output/xRNN"+args.token+"/output.tar.gz -C server_output/xRNN"+args.token+"/")