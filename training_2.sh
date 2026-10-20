#!/bin/bash
#SBATCH -p nvidia
#SBATCH -q nvidias
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=mof_reg_1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/model_shear_modulus_smallrad.out
#SBATCH --error=logs/model_shear_modulus_smallrad.err
 
echo "Job running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Load Singularity module
module load singularity/4.2.0

# Run inside the overlay and activate Conda environment
singularity exec --nv \
  --overlay /scratch/sk10275/AdsMOFNet-LIBRARY/AdsMOFNet.ext3 \
  /share/apps/admin/singularity-images/centos-8.2.2004.sif \
  /bin/bash -c "source /conda/miniforge3/bin/activate && conda activate /conda/miniforge3/envs/mof-dl-pointnet2-v2 && exec python /scratch/sk10275/PONY_ML/PONY_refactored_4/model_training_script_pbc_smallrad.py"