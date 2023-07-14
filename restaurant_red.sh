#!/bin/bash
####################
dataset='restaurants'
N=156
T=6
NT=$((N*T))
NT_array=($((NT/4)) $((NT/2)) $((NT)) $((NT*2)) $((NT*4)) $((NT*8)) $((NT*16)))
low=1
# high=85
high=35
# incr=$((8*2))
incr=$((8))
arr=( $(seq $low $incr $high) )
# for k in {5..50..$end}; do
for k in "${arr[@]}"; do
   for K in "${NT_array[@]}"; do
   #    # python reduce_main.py -k $k  -pr reach  -a greedymem -K $K -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_2.queries"  &
   #    # # python reduce_main.py -k $k  -pr reach  -a greedymem -K $K -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/restaurant_pair.true &
   #    # python reduce_main.py -k $k  -pr reach  -a greedymem -K $K -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_4.queries" &
   #    # # python reduce_main_crowd.py -k $k  -pr reach  -a greedymem -K $K -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -u c2 -cr data/large/crowd/restaurant_pair.true &
   #    # # python reduce_main.py -k $k  -pr reach  -a greedymem -K $K -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_5.queries" &
   #    #----
      python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_2.queries" -cr data/large/crowd/restaurant_pair.true -mq $mq -dh 0 &
      python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_2.queries" -cr data/large/crowd/restaurant_pair.true -u c2 -mq $mq -dh 0 &
      python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_4.queries" -cr data/large/crowd/restaurant_pair.true -mq $mq -dh 0 &
      python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_4.queries" -cr data/large/crowd/restaurant_pair.true -u c2 -mq $mq -dh 0 &
   #    #----
   done
   # python reduce_main_crowd.py -k $k  -pr reach  -a greedy -K $K -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_2.queries" -cr data/large/crowd/restaurant_pair.true -mq $mq -dh 0 &
   # python reduce_main_crowd.py -k $k  -pr reach  -a greedy -K $K -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_4.queries" -cr data/large/crowd/restaurant_pair.true -mq $mq -dh 0 &
done
#######################