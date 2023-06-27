#!/bin/bash
# for k in {2..4..1}; do
#    for i in {6..12..1}; do 
#       K=$((2**i)); 
#       dataset='papers'
#       python reduce_main_crowd.py -k $k  -pr reach  -a greedymem -K $K -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k &
#       python reduce_main_crowd.py -k $k  -pr reach  -a greedymem -K $K -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k &
#       python reduce_main_crowd.py -k $k  -pr reach  -a greedymem -K $K -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -u c2 &
#       python reduce_main_crowd.py -k $k  -pr reach  -a greedymem -K $K -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -u c2 &
#       dataset='products'
#       python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/product_pair.true &
#       python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/product_pair.true &
#       python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/product_pair.true -u c2 &
#       python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/product_pair.true -u c2
#    done
# done

# for k in {5..6..1}; do
#    for i in {6..12..1}; do 
#       K=$((2**i)); 
#       python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/product_pair.true &
#       python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/product_pair.true &
#       python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/product_pair.true -u c2 &
#       python reduce_main_crowd.py -k $k -K $K -pr reach -a greedymem -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -cr data/large/crowd/product_pair.true -u c2
#    done
# done
# for k in {1..5}; do
#    python reduce_main.py -k $k  -pr reach -a exact -d ER_15_22 -q data/queries/ER/ER_15_22_2.queries &
#    python reduce_main.py -k $k  -pr reach -a exact -d ER_15_22 -q data/queries/ER/ER_15_22_4.queries &
# done

for k in {2..4..1}; do
   for i in {6..12..1}; do 
      K=$((2**i)); 
      dataset='restaurants'
      python reduce_main_crowd.py -k $k  -pr reach  -a greedymem -K $K -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k &
      python reduce_main_crowd.py -k $k  -pr reach  -a greedymem -K $K -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k &
      python reduce_main_crowd.py -k $k  -pr reach  -a greedymem -K $K -ea mcbfs -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -u c2 &
      python reduce_main_crowd.py -k $k  -pr reach  -a greedymem -K $K -ea appr -d $dataset -q "data/queries/"$dataset"/"$dataset"_"$k".queries" -dh $k -u c2 &
   done
done