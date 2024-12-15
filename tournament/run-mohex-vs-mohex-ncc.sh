#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1

#SBATCH -p "cpu"
#SBATCH --qos="debug"
#SBATCH -t 00-00:30:00
source /etc/profile

./run-mohex-vs-mohex.sh --openings "openings/6x6-all-1ply" --size 6 --rounds 1 "mohex-simple" "mohex-simple"
