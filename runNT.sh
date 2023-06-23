#!/bin/bash
dataset='ER_15_22'
prefix='data/queries/ER/ER_15_22_'
opt_T_reach=11
opt_N_reach=85
opt_T_sp=165
opt_N_sp=26
opt_N_tri=6
opt_T_tri=100
declare -a trial=("appr" "mcapproxtri" "exact")
declare -a reachal=("appr" "eappr" "mcbfs" "pTmcbfs"  "rss" "pTrss" "exact")
declare -a spal=("exact" "appr" "eappr" "mcdij" "pTmcdij")
## reach
for al in "${reachal[@]}" 
do
   python measure_main_PeakMem.py -a $al -N $opt_N_reach -T $opt_T_reach -pr reach -d $dataset -q $prefix"1.queries" -S &
   python measure_main_PeakMem.py -a $al -N $opt_N_reach -T $opt_T_reach -pr reach -d $dataset -q $prefix"2.queries" -S &
   python measure_main_PeakMem.py -a $al -N $opt_N_reach -T $opt_T_reach -pr reach -d $dataset -q $prefix"3.queries" -S &
   python measure_main_PeakMem.py -a $al -N $opt_N_reach -T $opt_T_reach -pr reach -d $dataset -q $prefix"4.queries" -S 
done

### sp
for al in "${spal[@]}" 
do
   python measure_main_PeakMem.py -a $al -N $opt_N_sp -T $opt_T_sp -pr sp -d $dataset -q $prefix"1.queries" -S &
   python measure_main_PeakMem.py -a $al -N $opt_N_sp -T $opt_T_sp -pr sp -d $dataset -q $prefix"2.queries" -S &
   python measure_main_PeakMem.py -a $al -N $opt_N_sp -T $opt_T_sp -pr sp -d $dataset -q $prefix"3.queries" -S &
   python measure_main_PeakMem.py -a $al -N $opt_N_sp -T $opt_T_sp -pr sp -d $dataset -q $prefix"4.queries" -S
done

### tri
for al in "${trial[@]}" 
do
    python measure_main_PeakMem.py -a $al -N $opt_N_tri -T $opt_T_tri -pr tri -d $dataset -q $prefix"1.queries" -S
done
