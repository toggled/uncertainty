#!/bin/bash
# for k in {2..4..1}; do
#    for i in {6..12..1}; do 
#       K=$((2**i)); 
#       dataset='papers'
#       python reduce_main_crowd.py -k $k  -pr reach  -a greedymem -K $K -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k &
#       # python reduce_main_crowd.py -k $k  -pr reach  -a greedymem -K $K -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k &
#       python reduce_main_crowd.py -k $k  -pr reach  -a greedymem -K $K -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -u c2 &
#       # python reduce_main_crowd.py -k $k  -pr reach  -a greedymem -K $K -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -u c2 &
#       dataset='products'
#       python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/product_pair.true &
#       # python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/product_pair.true &
#       python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/product_pair.true -u c2 &
#       # python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/product_pair.true -u c2
#    done
# done

# dataset='products'
# for k in {5..6..1}; do
#    for i in {6..12..1}; do 
#       K=$((2**i)); 
#       python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/product_pair.true &
#       python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/product_pair.true &
#       python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/product_pair.true -u c2 &
#       python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/product_pair.true -u c2
#    done
# done

# for k in {2..4..1}; do
#    for i in {6..12..1}; do 
#       K=$((2**i)); 
#       dataset='restaurants'
#       python reduce_main_crowd.py -k $k  -pr reach  -a greedymem -K $K -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/restaurant_pair.true &
#       # python reduce_main_crowd.py -k $k  -pr reach  -a greedymem -K $K -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/restaurant_pair.true &
#       python reduce_main_crowd.py -k $k  -pr reach  -a greedymem -K $K -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -u c2 -cr data/large/crowd/restaurant_pair.true &
#       # python reduce_main_crowd.py -k $k  -pr reach  -a greedymem -K $K -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -u c2 -cr data/large/crowd/restaurant_pair.true &
#    done
# done

dataset='papers'
N=71
T=10
low=1
high=1527
# incr=$((152))
incr=152
budget_arr=( $(seq $low $incr $high) )
NT=$((N*T))
NT_array=($((NT)) $((NT*2)) $((NT*4)) $((NT*8)) $((NT*16)) $((NT*32)) $((NT*64)))
for k in "${budget_arr[@]}"; do
   # K=$((2**i)); 
   for K in "${NT_array[@]}"; do
      python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_2.queries" -dh 2 -cr data/large/crowd/paper_pair.true &
      # python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_4.queries" -dh 2 -cr data/large/crowd/paper_pair.true &
      python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_2.queries" -dh 2 -cr data/large/crowd/paper_pair.true -u c2 &
      # python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_4.queries" -dh 2 -cr data/large/crowd/paper_pair.true -u c2
   done
done