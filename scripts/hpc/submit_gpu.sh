#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process" 
### -- set the job Name -- 
#BSUB -J example_job
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=2GB]"
### -- set walltime limit: hh:mm -- 
#BSUB -W 00:05
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s194323@student.dtu.dk
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Error_%J.err 

# here follow the commands you want to execute
module load cuda/11.6
source ~/bachelor/bin/activate
python3 Desktop/Bachelor-project/Bachelor-project/scripts/hpc/example_gpu.py
