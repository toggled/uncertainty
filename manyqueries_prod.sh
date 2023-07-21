dataset='products'
N=71
T=10
low=1
high=1527
# incr=$((152))
incr=152
mq=-1
# budget_arr=( $(seq $low $incr $high) )
q_arr=(1 2 3 4 5 6 7 8 9)
k=100
for q in "${q_arr[@]}"; do
   # K=$((2**i)); 
   for r in "${q_arr[@]}"; do
      python reduce_main_crowd.py -k $k -pr reach -a greedyp -ea mcbfs -d $dataset -q "data/queries100/"$dataset"/"$dataset"_"$q".queries" -dh 0 -cr data/large/crowd/product_pair.true -mq $mq -r $r &
      python reduce_main.py -k $k -pr reach -a greedyp -ea mcbfs -d $dataset -q "data/queries100/"$dataset"/"$dataset"_"$q".queries" -mq $mq -r $r 
      python reduce_main_crowd.py -k $k -pr reach -a greedyp -ea mcbfs -d $dataset -q "data/queries100/"$dataset"/"$dataset"_"$q".queries" -dh 0 -cr data/large/crowd/product_pair.true -u c2 -mq $mq -r $r &
   done
done