#!/bin/bash
#SBATCH --job-name=csdi_physio_job   # Job name
#SBATCH --ntasks=1                   
#SBATCH --mem=16gb                   
#SBATCH --gres=gpu:2                 
#SBATCH --partition=gpu-v100         
#SBATCH --time=10:00:00              
#SBATCH --output=pm_%j.log  

# Activate the virtual environment
source /data/projects/graham.bishop/csdi_env/bin/activate  

# Navigate to the project directory
cd /data/projects/graham.bishop/CSDIBN

# Run training and imputation for the healthcare dataset X times
for i in {1..5}
do
    echo "Run $i"
    python exe_pm25.py --nsample 100
done

deactivates