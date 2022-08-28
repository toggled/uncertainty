# echo "Rank is: ${OMPI_COMM_WORLD_RANK}"

ulimit -t unlimited
shopt -s nullglob
# numthreads=$((OMPI_COMM_WORLD_SIZE))
mythread=$((OMPI_COMM_WORLD_RANK))

# echo $numthreads
# echo $mythread

# tlimit="2000"
# memlimit="4000000"
# ulimit -v $memlimit
# ulimit -v unlimited
ulimit -v 16000000



# python -u rundistribution.py -q reach --thread 16 > output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1 
# python -u rundistribution.py -q tri --thread 8 > output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1 
# python -u rundistribution.py -q diam --thread 8 > output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1 

python -u measure_rundistribution.py -q tri --thread 8 > output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1 

# wait
# multithread single iteration
# python -u distribution.py --iter 1 --thread $mythread --max_thread $numthreads > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1


# SIS Run
# python -u distribution_SIS.py --thread $mythread --max_thread $numthreads > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1

# single thread
# python -u distribution.py > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1

# multithread distribution_test.py single iteration
# python -u distribution_test.py --iter 1 --thread $mythread --max_thread $numthreads > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1

# multithread scalability experiments
# python -u distribution.py --scal --thread $mythread --max_thread $numthreads > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1

# kill $(ps aux | grep 'NNdhUiT' | grep 'rundistribution.py' | awk '{print $2}')
# kill $(ps aux | grep 'NNdhUiT' | grep 'main.py' | awk '{print $2}')
# grep -v "tri" results_k1.csv > res_k1.csv
# grep -v "tri" results_k2.csv > res_k2.csv
# grep -v "tri" results_k3.csv > res_k3.csv
# grep -v "tri" results_k4.csv > res_k4.csv
# grep -v "tri" results_k5.csv > res_k5.csv
# grep -v "tri" results_k6.csv > res_k6.csv
