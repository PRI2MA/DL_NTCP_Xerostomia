#!/bin/bash

#SBATCH --job-name=Xerostomia
#SBATCH --mail-type=END
#SBATCH --mail-user=d.h.chu@rug.nl
#SBATCH --time=23:59:59
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.log


# Install:
# module purge
# module load fosscuda/2020b
# module load OpenCV/4.2.0-foss-2020a-Python-3.8.2-contrib
# module load Python/3.8.6-GCCcore-10.2.0
# python3 -m venv /data/$USER/.envs/xerostomia_38
# source /data/$USER/.envs/xerostomia_38/bin/activate
# # pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# pip3 install pyparsing six python-dateutil
# pip3 install torchinfo tqdm monai pytz SimpleITK pydicom scikit-image matplotlib
# pip3 install torch_optimizer
# pip3 install scikit-learn
# pip3 install timm



# Run
module purge
module load fosscuda/2020b
# module load Python/3.7.4-GCCcore-8.3.0
# module load OpenCV/4.2.0-fosscuda-2019b-Python-3.7.4
module load OpenCV/4.2.0-foss-2020a-Python-3.8.2-contrib
module load Python/3.8.6-GCCcore-10.2.0
# module load PyTorch/1.10.0-fosscuda-2020b
## Activate local python environment
source /data/$USER/.envs/xerostomia_38/bin/activate

# Train
python3 main.py

