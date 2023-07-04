#!/bin/bash
N=11
T=85
NT=$((N*T))
NT_array=($((NT/4)) $((NT/2)) $((NT)) $((NT*2)) $((NT*3)) $((NT*4)) $((NT*5)) $((NT*6)))
for k in {1..5}; do
   # for i in {0..6}; do 
      # K=$((2**i)); 
   for K in "${NT_array[@]}"; do
      python reduce_main.py -k $k  -pr reach  -a greedymem -K $K -ea mcbfs -d ER_15_22 -q data/queries/ER/ER_15_22_2.queries &
      # python reduce_main.py -k $k  -pr reach  -a greedymem -K $K -ea exact -d ER_15_22 -q data/queries/ER/ER_15_22_2.queries &
      python reduce_main.py -k $k  -pr reach  -a greedymem -K $K -ea mcbfs -d ER_15_22 -q data/queries/ER/ER_15_22_4.queries &
      # python reduce_main.py -k $k  -pr reach  -a greedymem -K $K -ea exact -d ER_15_22 -q data/queries/ER/ER_15_22_4.queries &
      python reduce_main.py -k $k -K $K -pr reach -a greedy -ea mcbfs -d ER_15_22 -q data/queries/ER/ER_15_22_2.queries &
      # python reduce_main.py -k $k -K $K -pr reach -a greedy -ea exact -d ER_15_22 -q data/queries/ER/ER_15_22_2.queries &
      python reduce_main.py -k $k -K $K -pr reach -a greedy -ea mcbfs -d ER_15_22 -q data/queries/ER/ER_15_22_4.queries &
      # python reduce_main.py -k $k -K $K -pr reach -a greedy -ea exact -d ER_15_22 -q data/queries/ER/ER_15_22_4.queries 
   done
done

# for k in {1..5}; do
#    python reduce_main.py -k $k  -pr reach -a exact -d ER_15_22 -q data/queries/ER/ER_15_22_2.queries &
#    python reduce_main.py -k $k  -pr reach -a exact -d ER_15_22 -q data/queries/ER/ER_15_22_4.queries &
# done