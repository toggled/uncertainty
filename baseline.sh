#!/bin/bash
pr='sp'
N=286
T=11
data="products"
mq=-1
q="data/queries/products/products_5.queries"
cr="data/large/crowd/product_pair.true"
maxd=5
for budget in {5..5..25}; do
    python AlgoTKDE17.py -d $data -mq $mq -q $q -cr $cr -maxd $maxd -b $budget &
done