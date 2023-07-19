#!/bin/bash
N=161
T=6
ea="exact"
NT=$((N*T))
# NT_array=($((NT*NT)) $((4*NT*NT)) $((8*NT*NT)))
NT_array=($((NT/4)) $((NT/2)) $((NT)) $((NT*2)) $((NT*4)) $((NT*8)) $((NT*16)))
# NT_array=($((NT/4)) $((NT/2)) $((NT)) $((NT*2)) $((NT*3)) $((NT*4)) $((NT*5)) $((NT*6)))
for k in {2..2}; do
   # for i in {0..6}; do 
      # K=$((2**i)); 
   # for K in "${NT_array[@]}"; do
   #    python reduce_main.py -k $k  -pr reach  -a greedymem -K $K -ea $ea -d ER_15_22 -q data/queries/ER/ER_15_22_2.queries &
   #    # python reduce_main.py -k $k  -pr reach  -a greedymem -K $K -ea exact -d ER_15_22 -q data/queries/ER/ER_15_22_2.queries &
   #    # python reduce_main.py -k $k  -pr reach  -a greedymem -K $K -ea $ea -d ER_15_22 -q data/queries/ER/ER_15_22_4.queries &
   #    # python reduce_main.py -k $k  -pr reach  -a greedymem -K $K -ea exact -d ER_15_22 -q data/queries/ER/ER_15_22_4.queries &
   #    # python reduce_main.py -k $k -K $K -pr reach -a greedy -ea mcbfs -d ER_15_22 -q data/queries/ER/ER_15_22_2.queries &
   #    # python reduce_main.py -k $k -K $K -pr reach -a greedy -ea exact -d ER_15_22 -q data/queries/ER/ER_15_22_2.queries &
   #    # python reduce_main.py -k $k -K $K -pr reach -a greedy -ea mcbfs -d ER_15_22 -q data/queries/ER/ER_15_22_4.queries &
   #    # python reduce_main.py -k $k -K $K -pr reach -a greedy -ea exact -d ER_15_22 -q data/queries/ER/ER_15_22_4.queries 
   # done
   python reduce_main.py -k $k -pr reach -a greedyp -ea $ea -d ER_15_22 -q data/queries/ER/ER_15_22_2.queries &
   # python reduce_main.py -k $k -pr reach -a greedy -ea $ea -d ER_15_22 -q data/queries/ER/ER_15_22_4.queries &
done
# for K in "${NT_array[@]}"; do
#    # python reduce_main.py -k $k  -pr reach  -a greedymem -K $K -ea exact -d ER_15_22 -q data/queries/ER/ER_15_22_2.queries &
#    python reduce_main.py -k 4  -pr reach  -a greedymem -K $K -ea $ea -d ER_15_22 -q data/queries/ER/ER_15_22_4.queries &
# done
python reduce_main.py -k 4 -pr reach -a greedyp -ea $ea -d ER_15_22 -q data/queries/ER/ER_15_22_4.queries &
# python reduce_main.py -k 4  -pr reach -a exact -d ER_15_22 -q data/queries/ER/ER_15_22_4.queries &
# for k in {1..2}; do
#    python reduce_main.py -k $k  -pr reach -a exact -d ER_15_22 -q data/queries/ER/ER_15_22_2.queries &
#    python reduce_main.py -k $k  -pr reach -a exact -d ER_15_22 -q data/queries/ER/ER_15_22_4.queries &
# done