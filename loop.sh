#!/bin/bash
# ONLY FOR BIOMINE, FLICKR (And large graphs in general)
FROMHERE=701
ENDHERE=301
delta=100
for ((i=FROMHERE; i>=ENDHERE; i-=delta))
do
     #python measure_main.py -a appr -pre 1 -pr tri -T $i # ER
	 srun --mem 256G singularity exec pytorch_512G.sif python3 measure_main.py -a eappr -N 1 -T $i -pr tri -d flickr -q data/queries/flickr/flickr_1.queries -pre 1
	 
	 # srun --cpus-per-task 5 --mem 256G singularity exec pytorch_512G.sif python3 measure_main.py -a eappr -N 1 -T $i -pr reach -d biomine -q data/queries/biomine/biomine_1.queries -pre 1
done
