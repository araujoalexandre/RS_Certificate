#!/usr/bin/bash
#SBATCH --account=yxj@gpu
#SBATCH --job-name=cert
#SBATCH --time=120
#SBATCH --partition=gpu_p13
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --ntasks-per-node=40
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --output=/linkhome/rech/genlam01/uuc79vj/rs_certificat/slurm_log-%j.out
#SBATCH --error=/linkhome/rech/genlam01/uuc79vj/rs_certificat/slurm_log-%j.err
#SBATCH --exclusive

module purge
module load pytorch-gpu/py3/1.8.0
unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY;

sigma0=$1
sigma1=$2
pABar_sigma0=$3
pABar_sigma1=$4

srun --output=log_${sigma0}_${sigma1}_${pABar_sigma0}_${pABar_sigma1}.logs --mpi=pmi2 \
  python3 -u certify_multi_sigma.py \
  --sigma0 ${sigma0} \
  --sigma1 ${sigma1} \
  --pABar_sigma0 ${pABar_sigma0} \
  --pABar_sigma1 ${pABar_sigma1} 


