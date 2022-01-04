
import argparse
import time
import torch
import numpy as np
import vegas
# from mpi4py import MPI
from numpy import exp, log, sqrt
from scipy.stats import norm
from scipy.special import erfinv
from torch.distributions.log_normal import LogNormal


class Certificate:
  """ Compute certificate from generalized Neyman-Pearson Lemma"""
  
  def __init__(self, dist, sigma0, sigma1, pABar_sigma0, pABar_sigma1, dim,
               nitn, neval, nitn_train=None, neval_train=None, verbose=False):

    self.dist = dist
    # self.rank = dist.Get_rank()
    self.rank = 0
    self.verbose = verbose
 
    self.sigma0 = sigma0
    self.sigma1 = sigma1
    self.pABar_sigma0 = pABar_sigma0
    self.pABar_sigma1 = pABar_sigma1
    
    self.dim = dim
    
    self.nitn = nitn
    self.neval = neval
    self.nitn_train = nitn_train
    self.neval_train = neval_train
    if self.nitn_train is None:
      self.nitn_train = nitn
    if self.neval_train is None:
      self.neval_train = neval

    self.percentile = 1 - 1e-16

    # binary search parameters
    self.bs_cert_tol = 0.001
    self.bs_k_values_close_tol = 0.002
    self.is_close_bellow = lambda x, y: y - self.bs_k_values_close_tol <= x <= y
    
    # Values from Cohen et al.
    self.cert_cohen = self.sigma0 * norm.ppf(self.pABar_sigma0)
    if self.rank == 0:
      print(f'Certificat Cohen et al.: {self.cert_cohen}')

  def _debug_print(self, msg):
    if self.rank == 0 and self.verbose:
      print(msg)
    
  def compute_k1_cohen(self, eps):
    """ Compute the constant k1 in the context of Cohen certificate """
    return exp((eps / self.sigma0 * norm.ppf(self.pABar_sigma0)) - eps**2 / (2 * self.sigma0**2))
    
  def check_better_certificate(self):
    """ Check if a better certificate is possible """
    k1_cohen = self.compute_k1_cohen(self.eps)
    p0, p1 = self.compute_p0_p1(k1_cohen, 0)
    print('k1_cohen', k1_cohen)
    print(f'p0: {p0}, p1: {p1}')
    if p1 <= pABar_sigma1:
      return True
    else:
      return False

  def lognormal_icdf(self, x, mu, sigma):
    return exp(mu + sqrt(2) * sigma * erfinv(2*x - 1)) 

  def b(self, k1):
    return -self.eps**2 / (2*self.sigma0**2) - log(k1)

  def define_distributions0(self, k1, log_k2):
    self.a1 = (self.sigma0*self.eps) / self.sigma0**2
    self.c1 = +1/2 * (1 - self.sigma0**2 / self.sigma1**2)
    mu, sigma = self.b(k1), self.a1
    U = LogNormal(mu, sigma)
    m1 = self.lognormal_icdf(self.percentile, mu, sigma)
    if log_k2 == 0:
      return U, None, m1, None
    mu = self.c1 * self.dim + log_k2 - log(k1)
    sigma = self.c1*sqrt(2*self.dim)
    R = LogNormal(mu, sigma)
    m2 = self.lognormal_icdf(self.percentile, mu, sigma)
    return U, R, m1, m2

  def define_distributions1(self, k1, log_k2):
    self.a2 = (self.sigma1*self.eps) / self.sigma0**2
    self.c2 = -1/2 * (1 - self.sigma1**2 / self.sigma0**2)
    mu, sigma = self.b(k1), self.a2
    U = LogNormal(mu, sigma)
    m1 = self.lognormal_icdf(self.percentile, mu, sigma)
    if log_k2 == 0:
      return U, None, m1, None
    mu, sigma = self.c2 * self.dim + log_k2 - log(k1), self.c2*sqrt(2*self.dim)
    R = LogNormal(mu, sigma)
    m2 = self.lognormal_icdf(self.percentile, mu, sigma)
    return U, R, m1, m2
  
  def define_distributions_certificate(self, k1, log_k2):
    ratio = 1/(self.sigma0**2) - 1/(self.sigma1**2)
    self.a1 = (self.sigma0 * self.eps) / (self.sigma0**2)
    self.b1 = - self.eps**2 / (2*self.sigma0**2) + log(k1)
    self.a2 = 1/2 * self.sigma0**2 * ratio * sqrt(2*self.dim)
    self.b2 = -self.eps**2/(2*self.sigma1**2) + log_k2 + 1/2 * self.sigma0**2 * ratio * self.dim
    self.a3 = 1/2 * self.sigma0**2 * ratio
    self.a4 = (-self.sigma0 * self.eps)/(self.sigma1**2)
    U = LogNormal(0, self.a1)
    m1 = self.lognormal_icdf(self.percentile, 0, self.a1)
    R = LogNormal(self.b2, self.a2)
    m2 = self.lognormal_icdf(self.percentile, self.b2, self.a2)
    return U, R, m1, m2

  @vegas.batchintegrand
  def batch_integrand(self, x):
    x = torch.Tensor(x).to(f'cuda:{self.rank}')
    x, y = x[:, 0], x[:, 1]
    x_pdf = torch.exp(self.U.log_prob(x))
    y_pdf = torch.exp(self.R.log_prob(y))
    res = x_pdf * y_pdf * ((x - y * x**(2*self.c1/self.a1) <= 1.)*1.)
    return np.double(res.cpu().numpy())

  @vegas.batchintegrand
  def batch_integrand_certificate(self, x):
    x = torch.Tensor(x).to(f'cuda:{self.rank}')
    x, y = x[:, 0], x[:, 1]
    x_pdf = torch.exp(self.U.log_prob(x))
    y_pdf = torch.exp(self.R.log_prob(y))
    res = x_pdf * y_pdf * (x * exp(self.b1) + y * x**(2*self.a3/self.a2) * x**(self.a4/self.a2) >= 1.)
    return np.double(res.cpu().numpy())

  def compute_integral(self, U, R, k1, log_k2, support):
    self.U, self.R = U, R
    if log_k2 == 0: # cohen certificate
      result = U.cdf(torch.Tensor([1.]))
    else:
      integ = vegas.Integrator([[0, support[0]], [0, support[1]]])
      integ(self.batch_integrand, nitn=self.nitn, neval=self.neval)
      result = integ(self.batch_integrand, nitn=self.nitn, neval=self.neval).mean
    return result
  
  def compute_certificate(self, k1, log_k2):
    if log_k2 == 0: # cohen certificate
      a1 = (self.sigma0 * self.eps) / (self.sigma0**2)
      b1 = - self.eps**2 / (2*self.sigma0**2) + log(k1)
      U = LogNormal(b1, a1)
      result = 1 - U.cdf(torch.Tensor([1.]))
    else:
      self.U, self.R, *support = self.define_distributions_certificate(k1, log_k2)
      integ = vegas.Integrator([[0, support[0]], [0, support[1]]])
      integ(self.batch_integrand_certificate, nitn=self.nitn, neval=self.neval)
      result = integ(self.batch_integrand_certificate, nitn=self.nitn, neval=self.neval).mean
    return result
  
  def compute_p0_p1(self, *args):
    return self.compute_p0(*args), self.compute_p1(*args)
  
  def compute_p0(self, *args):
    k1, log_k2 = args
    U, R, *support = self.define_distributions0(k1, log_k2)
    return self.compute_integral(U, R, k1, log_k2, support)

  def compute_p1(self, *args):
    k1, log_k2 = args
    U, R, *support = self.define_distributions1(k1, log_k2)
    return self.compute_integral(U, R, k1, log_k2, support)

  def adjust_k1(self, k1, log_k2):
    start = 0.
    end = k1 + 0.1
    self._debug_print('\n-- Adjust k1 --')
    while True:
      k1 = (start + end) / 2
      U0, R0, *support = self.define_distributions0(k1, log_k2)
      p0 = self.compute_integral(U0, R0, k1, log_k2, support)
      self._debug_print(f'start {start}, end {end}, p0 {p0:.5f} / {self.pABar_sigma0}')
      if self.is_close_bellow(p0, self.pABar_sigma0):
        break
      if p0 < self.pABar_sigma0:
        start = k1
      else:
        end = k1
    return k1, log_k2, p0
  
  def adjust_k2(self, k1, log_k2):
    start = 0.
    end = -(self.c2 * self.dim + 50)
    self._debug_print('\n-- Adjust k2 --')
    while True:
      log_k2 = (start + end) / 2
      U1, R1, *support = self.define_distributions1(k1, log_k2)
      p1 = self.compute_integral(U1, R1, k1, log_k2, support)
      self._debug_print(f'start {start}, end {end}, p1 {p1:.5f} / {self.pABar_sigma1}')
      if self.is_close_bellow(p1, self.pABar_sigma1):
        break
      if p1 > self.pABar_sigma1:
        start = log_k2
      else:
        end = log_k2
    return k1, log_k2, p1
  
  def find_k_values(self):
    k1 = self.compute_k1_cohen(self.eps)
    log_k2 = 0
    p0, p1 = self.compute_p0_p1(k1, log_k2)
    while True:
      k1, log_k2, p1 = self.adjust_k2(k1, log_k2)
      p0 = self.compute_p0(k1, log_k2)
      if self.is_close_bellow(p0, self.pABar_sigma0) and self.is_close_bellow(p1, self.pABar_sigma1):
        break
      k1, log_k2, p0 = self.adjust_k1(k1, log_k2)
      p1 = self.compute_p1(k1, log_k2)
      if self.is_close_bellow(p0, self.pABar_sigma0) and self.is_close_bellow(p1, self.pABar_sigma1):
        break
    return k1, log_k2

  def check_certificate(self, eps):
    self.eps = eps
    k1, log_k2 = self.find_k_values()
    cert = self.compute_certificate(k1, log_k2)
    return cert >= 0.5

  def best_certificate(self):
    start = self.cert_cohen - 0.01
    end = start + 0.15
    while (end - start) >= self.bs_cert_tol:
      self._debug_print(f'start: {start:.4f}, end: {end:.4f}')
      eps = (start + end) / 2
      if self.check_certificate(eps):
        start = eps
      else:
        end = eps
    return start



if __name__ == '__main__':

  parser = argparse.ArgumentParser(
      description='Compute Randomized Smoothing Certificate.')
  parser.add_argument("--dim", type=int, help="Dimension.")
  args = parser.parse_args()

  # dist = MPI.COMM_WORLD
  # size = dist.Get_size()
  # rank = dist.Get_rank()
  rank = 0

  sigma0 = 0.25
  sigma1 = 0.25
  pABar_sigma0 = 0.90
  pABar_sigma1 = 0.90
  dim = args.dim
  nitn = 60
  neval = 4000000
  verbose = True

  if rank == 0:
    print('-- Context --')
    print(f'sigma0 {sigma0}, sigma1 {sigma1}')
    print(f'pABar_sigma0 {pABar_sigma0}, pABar_sigma1 {pABar_sigma1}')
    print(f'Dim: {dim}')

  cert = Certificate(None, sigma0, sigma1, pABar_sigma0, pABar_sigma1, dim-2,
                     nitn, neval, nitn_train=nitn, neval_train=neval, verbose=verbose)

  cert.eps = 0.2
  print(cert.eps, cert.check_better_certificate())

  cert.eps = 0.5
  print(cert.eps, cert.check_better_certificate())

  cert.eps = 0.75
  print(cert.eps, cert.check_better_certificate())

  # start = time.time()
  # best_cert = cert.best_certificate()
  # if rank == 0:
  #   print(f'Dim {dim}, best certificate is {best_cert}, duration: {time.time() - start:.3f}')

  # start_global = time.time()
  # for eps in np.arange(0.25, 3.25, 0.25):
  #   start = time.time()
  #   certificate = cert.check_certificate(eps)
  #   if rank == 0:
  #     print(f'Certification at {eps}: {certificate}, duration {time.time() - start:.3f}')
  # print(f'global duration: {time.time() - start_global:.3f}')



