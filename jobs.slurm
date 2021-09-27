#!/usr/bin/bash
#SBATCH --account=yxj@gpu
#SBATCH --job-name=cert
#SBATCH --time=1200
#SBATCH --partition=gpu_p13
#SBATCH --ntasks=40
#SBATCH --ntasks-per-node=40
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --output=/linkhome/rech/genlam01/uuc79vj/rs_certificat/slurm_log-%j.out
#SBATCH --error=/linkhome/rech/genlam01/uuc79vj/rs_certificat/slurm_log-%j.err
#SBATCH --exclusive

module purge
module load pytorch-gpu/py3/1.9.0
unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY;
export CUDA_VISIBLE_DEVICES=''

dim=$1

srun --output=log_certificate_${dim}.logs python3 -u certificat.py --dim ${dim}


