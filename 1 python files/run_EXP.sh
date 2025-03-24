#!/bin/bash
#SBATCH -n 4
#SBATCH --nodes=1
#SBATCH --job-name=exp
#SBATCH --output=exp.out
#SBATCH --error=exp.err
#SBATCH --mem-per-cpu=16000
#SBATCH --time=144:00:00         

# load some modules & list loaded modules
module load stack/2024-06 python/3.11.6

pip3 install --upgrade pip

#install relevant libraries
pip3 install numpy scipy matplotlib ipyparallel scikit-build
pip3 install --upgrade ngsolve 

rm -f exp.data

# echo "=================== d=2, l=0 =======================" | tee -a exp.data
# python3 d2l0_study.py | tee -a exp.data
# echo "=================== d=3, l=0 =======================" | tee -a exp.data
# python3 d3l0_study.py | tee -a exp.data

# echo "=================== d=2, l=1 =======================" | tee -a exp.data
# python3 d2l1_study.py | tee -a exp.data

# echo "=================== d=2, l=2 =======================" | tee -a exp.data
# python3 d2l2_study.py | tee -a exp.data
# echo "=================== d=3, l=3 =======================" | tee -a exp.data
# python3 d3l3_study.py | tee -a exp.data

echo "=================== d=3, l=1 =======================" | tee -a exp.data
python3 d3l1_study.py | tee -a exp.data
echo "=================== d=3, l=2 =======================" | tee -a exp.data
python3 d3l2_study.py | tee -a exp.data







