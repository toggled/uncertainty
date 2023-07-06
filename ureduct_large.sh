#!/bin/bash
# for k in {5..50..5}; do
#    # for i in {6..12..1}; do 
#    #    K=$((2**i)); 
#    for K in "${NT_array[@]}"; do
#       dataset='papers'
#       python reduce_main_crowd.py -k $k  -pr reach  -a greedymem -K $K -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_2.queries" &
#       # python reduce_main_crowd.py -k $k  -pr reach  -a greedymem -K $K -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k &
#       python reduce_main_crowd.py -k $k  -pr reach  -a greedymem -K $K -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_4.queries" &
#       # python reduce_main_crowd.py -k $k  -pr reach  -a greedymem -K $K -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -u c2 &
#       # dataset='products'
#       # python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/product_pair.true &
#       # # python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/product_pair.true &
#       # python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/product_pair.true -u c2 &
#       # # python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/product_pair.true -u c2
#    done
# done
dataset='products'
N=46
T=4
NT=$((N*T))
NT_array=($((NT/4)) $((NT/2)) $((NT)) $((NT*2)) $((NT*4)) $((NT*8)) $((NT*16)) $((NT*32)))
low=1
high=376
incr=$((37*2))
arr=( $(seq $low $incr $high) )
# for k in {5..50..5}; do
for k in "${arr[@]}"; do
   # for i in {6..12..1}; do 
   #    K=$((2**i)); 
   for K in "${NT_array[@]}"; do
      python reduce_main.py -k $k -K $K -pr reach -a greedymem -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_2.queries" &
      # python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/product_pair.true &
      python reduce_main.py -k $k -K $K -pr reach -a greedymem -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_4.queries" &
      # python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/product_pair.true -u c2
      python reduce_main.py -k $k -K $K -pr reach -a greedymem -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_6.queries" &
   done
   python reduce_main.py -k $k -K $K -pr reach -a greedy -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_2.queries" &
   # python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/product_pair.true &
   python reduce_main.py -k $k -K $K -pr reach -a greedy -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_4.queries" &
   # python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/product_pair.true -u c2
   python reduce_main.py -k $k -K $K -pr reach -a greedy -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_6.queries" &

done

dataset='restaurants'
N=156
T=6
NT=$((N*T))
NT_array=($((NT/4)) $((NT/2)) $((NT)) $((NT*2)) $((NT*4)) $((NT*8)) $((NT*16)) $((NT*32)))
low=1
high=851
incr=$((85*2))
arr=( $(seq $low $incr $high) )
# for k in {5..50..$end}; do
for k in "${arr[@]}"; do
   for K in "${NT_array[@]}"; do
      python reduce_main.py -k $k  -pr reach  -a greedymem -K $K -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_2.queries"  &
      # python reduce_main_crowd.py -k $k  -pr reach  -a greedymem -K $K -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/restaurant_pair.true &
      python reduce_main.py -k $k  -pr reach  -a greedymem -K $K -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_4.queries" &
      # python reduce_main_crowd.py -k $k  -pr reach  -a greedymem -K $K -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -u c2 -cr data/large/crowd/restaurant_pair.true &
      # python reduce_main.py -k $k  -pr reach  -a greedymem -K $K -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_5.queries" &
   done
   python reduce_main.py -k $k  -pr reach  -a greedy -K $K -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_2.queries"  &
   python reduce_main.py -k $k  -pr reach  -a greedy -K $K -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_4.queries" &
done