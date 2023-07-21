dataset='products'
N=71
T=10
low=1
high=1527
# incr=$((152))
incr=152
mq=-1
# budget_arr=( $(seq $low $incr $high) )
q_arr=(1 2 3 4 5 6 7 8 9 10)
NT=$((N*T))
NT_array=($((NT)) $((NT*2)) $((NT*4)) $((NT*8)) $((NT*16)) $((NT*32)) $((NT*64)))
k=100
for q in "${q_arr[@]}"; do
   # K=$((2**i)); 
   for r in "${q_arr[@]}"; do
      python reduce_main_crowd.py -k $k -pr reach -a greedyp -ea mcbfs -d $dataset -q "data/queries100/"$dataset"/"$dataset"_"$q".queries" -dh 0 -cr data/large/crowd/product_pair.true -mq $mq -r $r &
      python reduce_main_crowd.py -k $k -pr reach -a greedyp -ea mcbfs -d $dataset -q "data/queries100/"$dataset"/"$dataset"_"$q".queries" -dh 0 -cr data/large/crowd/product_pair.true -u c2 -mq $mq -r $r &
   done
done