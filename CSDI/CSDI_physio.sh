#!/bin/bash
#SBATCH --job-name=csdi_physio_job   # Job name
#SBATCH --ntasks=1                   
#SBATCH --mem=16gb                   
#SBATCH --gres=gpu:2                 
#SBATCH --partition=gpu-v100         
#SBATCH --time=30:00:00              
#SBATCH --output=physio_.1%j.log  

# Activate the virtual environment
source /data/projects/graham.bishop/csdi_env/bin/activate  # Adjust the path to your environment

# Navigate to the project directory
cd /scratch/graham.bishop/CSDI

# Run training and imputation for the healthcare dataset 5 times
for i in {1..5}
do
    echo "Run $i"
    python exe_physio.py --testmissingratio 0.9 --nsample 100
done

# Deactivate the virtual environment (optional)
deactivates