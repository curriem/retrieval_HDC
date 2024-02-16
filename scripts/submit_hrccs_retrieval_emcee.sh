#!/bin/bash

#SBATCH --job-name=retrieval_emcee

#SBATCH --account=vsm
#SBATCH --partition=compute-hugemem

#SBATCH --nodes=1
#SBATCH --time=336:00:00
#SBATCH --mem=740G

#SBATCH --ntasks-per-node=40
#SBATCH --exclusive

#SBATCH --chdir=/gscratch/vsm/mcurr/PROJECTS/retrieval_HDC/scripts

ulimit -s unlimited

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/gscratch/vsm/mcurr/anaconda3_install/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/gscratch/vsm/mcurr/anaconda3_install/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/gscratch/vsm/mcurr/anaconda3_install/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/gscratch/vsm/mcurr/anaconda3_install/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate smrtr

python call_hrccs_retrieval.py emcee >  call_hrccs_retrieval_emcee.log
wait
