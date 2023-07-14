#!/bin/bash
####################
mq=10
dataset='restaurants'
N=156
T=6
NT=$((N*T))
NT_array=($((NT/4)) $((NT/2)) $((NT)) $((NT*2)) $((NT*4)))
low=5
high=25
incr=$((5))
arr=( $(seq $low $incr $high) )
# for k in {5..50..$end}; do
for k in "${arr[@]}"; do
   for K in "${NT_array[@]}"; do
      python reduce_main.py -k $k -K $K -pr reach -a greedymem -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_2.queries" -mq $mq &
      python reduce_main.py -k $k -K $K -pr reach -a greedymem -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_4.queries" -mq $mq &
   done
   python reduce_main.py -k $k  -pr reach  -a greedy -K $K -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_2.queries" -mq $mq &
   python reduce_main.py -k $k  -pr reach  -a greedy -K $K -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_4.queries" -mq $mq &
done
####################
dataset='products'
N=46
T=4
NT=$((N*T))
NT_array=($((NT/4)) $((NT/2)) $((NT)) $((NT*2)) $((NT*4)))
low=5
high=25
incr=$((5))
arr=( $(seq $low $incr $high) )
# for k in {5..50..$end}; do
for k in "${arr[@]}"; do
   for K in "${NT_array[@]}"; do
      python reduce_main.py -k $k -K $K -pr reach -a greedymem -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_2.queries" -mq $mq &
      python reduce_main.py -k $k -K $K -pr reach -a greedymem -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_4.queries" -mq $mq &
   done
   python reduce_main.py -k $k -pr reach -a greedy -K $K -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_2.queries" -mq $mq &
   python reduce_main.py -k $k -pr reach -a greedy -K $K -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_4.queries" -mq $mq &
done
#######################