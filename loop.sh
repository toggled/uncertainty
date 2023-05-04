#!/bin/bash

FROMHERE=701

for ((i=FROMHERE; i>=301; i-=100))
do
     python measure_main.py -a appr -pre 1 -pr tri -T $i
done