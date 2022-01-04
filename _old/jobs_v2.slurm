#!/usr/bin/bash
#SBATCH --account=yxj@gpu
#SBATCH --job-name=cert
#SBATCH --output=/linkhome/rech/genlam01/uuc79vj/rs_certificat/slurm_log-%j.out
#SBATCH --error=/linkhome/rech/genlam01/uuc79vj/rs_certificat/slurm_log-%j.err
#SBATCH --time=1200
#SBATCH --partition=gpu_p2
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --wait-all-nodes=1
#SBATCH --exclusive
module purge
module load pytorch-gpu/py3/1.9.0
unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY;

srun --output=/linkhome/rech/genlam01/uuc79vj/rs_certificat/log_certificate_v2.logs --unbuffered --open-mode=append --nodes=1 --gres=gpu:8 python3 /linkhome/rech/genlam01/uuc79vj/rs_certificat/certificat_v2.py

