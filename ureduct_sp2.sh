#!/bin/bash
pr='sp'
N=11
T=85
NT=$((N*T))
NT_array=($((NT/4)) $((NT/2)) $((NT)) $((NT*2)) $((NT*3)) $((NT*4)) $((NT*5)) $((NT*6)))
for k in {1..5}; do
   # for i in {0..6}; do 
   #    K=$((2**i)); 
   for K in "${NT_array[@]}"; do
      python reduce_main.py -k $k  -pr $pr  -a greedymem -K $K -ea mcdij -d ER_15_22 -q data/queries/ER/ER_15_22_2.queries &
      # python reduce_main.py -k $k  -pr $pr  -a greedymem -K $K -ea exact -d ER_15_22 -q data/queries/ER/ER_15_22_2.queries &
      python reduce_main.py -k $k  -pr $pr  -a greedymem -K $K -ea mcdij -d ER_15_22 -q data/queries/ER/ER_15_22_4.queries &
      # python reduce_main.py -k $k  -pr $pr  -a greedymem -K $K -ea exact -d ER_15_22 -q data/queries/ER/ER_15_22_4.queries &
      python reduce_main.py -k $k -K $K -pr $pr -a greedy -ea mcdij -d ER_15_22 -q data/queries/ER/ER_15_22_2.queries &
      # python reduce_main.py -k $k -K $K -pr $pr -a greedy -ea exact -d ER_15_22 -q data/queries/ER/ER_15_22_2.queries &
      python reduce_main.py -k $k -K $K -pr $pr -a greedy -ea mcdij -d ER_15_22 -q data/queries/ER/ER_15_22_4.queries &
      # python reduce_main.py -k $k -K $K -pr $pr -a greedy -ea exact -d ER_15_22 -q data/queries/ER/ER_15_22_4.queries 
   done
done
# for k in {1..5}; do
#    python reduce_main.py -k $k  -pr $pr -a exact -d ER_15_22 -q data/queries/ER/ER_15_22_2.queries &
#    python reduce_main.py -k $k  -pr $pr -a exact -d ER_15_22 -q data/queries/ER/ER_15_22_4.queries &
# done