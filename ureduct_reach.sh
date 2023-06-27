#!/bin/bash
for k in {1..5}; do
   for i in {0..6}; do 
      K=$((2**i)); 
      python reduce_main.py -k $k  -pr reach  -a greedymem -K $K -ea mcbfs -d ER_15_22 -q data/queries/ER/ER_15_22_2.queries &
      python reduce_main.py -k $k  -pr reach  -a greedymem -K $K -ea exact -d ER_15_22 -q data/queries/ER/ER_15_22_2.queries &
      python reduce_main.py -k $k  -pr reach  -a greedymem -K $K -ea mcbfs -d ER_15_22 -q data/queries/ER/ER_15_22_4.queries &
      python reduce_main.py -k $k  -pr reach  -a greedymem -K $K -ea exact -d ER_15_22 -q data/queries/ER/ER_15_22_4.queries 
   done
done
for k in {1..5}; do
   for i in {0..6}; do 
      K=$((2**i)); 
      python reduce_main.py -k $k -K $K -pr reach -a greedy -ea mcbfs -d ER_15_22 -q data/queries/ER/ER_15_22_2.queries &
      python reduce_main.py -k $k -K $K -pr reach -a greedy -ea exact -d ER_15_22 -q data/queries/ER/ER_15_22_2.queries &
      python reduce_main.py -k $k -K $K -pr reach -a greedy -ea mcbfs -d ER_15_22 -q data/queries/ER/ER_15_22_4.queries &
      python reduce_main.py -k $k -K $K -pr reach -a greedy -ea exact -d ER_15_22 -q data/queries/ER/ER_15_22_4.queries 
   done
done
for k in {1..5}; do
   python reduce_main.py -k $k  -pr reach -a exact -d ER_15_22 -q data/queries/ER/ER_15_22_2.queries &
   python reduce_main.py -k $k  -pr reach -a exact -d ER_15_22 -q data/queries/ER/ER_15_22_4.queries &
done