#!/bin/bash

#SBATCH -N 1
#SBATCH -c 8

#SBATCH -p "cpu"
#SBATCH --qos="short"
#SBATCH -t 00-00:10:00
source /etc/profile

./run-mohex-vs-mohex.sh --openings "openings/7x7-all-1ply" --size 7 --rounds 1 "7x7-settings" "7x7-settings"
