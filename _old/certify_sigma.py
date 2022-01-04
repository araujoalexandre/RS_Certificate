
import argparse
import time
# import torch
import numpy as np
import vegas
from numpy import exp, log, sqrt
from scipy.stats import norm, lognorm


class Integrand(vegas.BatchIntegrand):

  def __init__(self, U, R, a, c):
    self.U = U
    self.R = R
    self.a = a
    self.c = c

  def integrand(self, x, y):
    return self.U.pdf(x) * self.R.pdf(y) * ((x - y * x**(2*self.c/self.a) <= 1.)*1.)

  def __call__(self, t):
    x, y = t[:, 0], t[:, 1]
    t1 = (1 - x) / x
    t2 = (1 - y) / y
    return self.integrand(t1, t2) * 1/x**2 * 1/y**2



class IntegrandCertificate(vegas.BatchIntegrand):

  def __init__(self, U, R, b1, a2, a3, a4):                                                             
    self.U = U
    self.R = R
    self.b1 = b1
    self.a2 = a2
    self.a3 = a3
    self.a4 = a4

  def integrand(self, x, y):
    return self.U.pdf(x) * self.R.pdf(y) * (x * exp(self.b1) + y * x**(2*self.a3/self.a2) * x**(self.a4/self.a2) >= 1.)

  def __call__(self, t):
    x, y = t[:, 0], t[:, 1]
    t1 = (1 - x) / x
    t2 = (1 - y) / y
    return self.integrand(t1, t2) * 1/x**2 * 1/y**2




class Certificate:
  """ Compute certificate from generalized Neyman-Pearson Lemma"""
  
  def __init__(self, dist, sigma0, sigma1, pABar_sigma0, pABar_sigma1, dim,
               nitn, neval, nitn_train=None, neval_train=None, verbose=False):

    self.dist = dist
    self.rank = dist.Get_rank()
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
    self.eps = 0.1
    k1_cohen = self.compute_k1_cohen(self.eps)
    p0, p1 = self.compute_p0_p1(k1_cohen, 0)
    if p1 <= pABar_sigma1:
      return True
    else:
      return False

  def b(self, k1):
    return -self.eps**2 / (2*self.sigma0**2) - log(k1)

  def define_distributions0(self, k1, log_k2):
    a1 = (self.sigma0*self.eps) / self.sigma0**2
    c1 = +1/2 * (1 - self.sigma0**2 / self.sigma1**2)
    U = lognorm(np.abs(a1), loc=0, scale=exp(self.b(k1)))
    if log_k2 == 0:
      return U, None, a1, c1
    # self._debug_print(f'scale: {self.c1*sqrt(2*self.dim)}')
    # self._debug_print(f'mu: {self.c1 * self.dim + log_k2 - log(k1)}')
    R = lognorm(np.abs(c1*sqrt(2*self.dim)), loc=0, scale=exp(c1 * self.dim + log_k2 - log(k1)))
    return U, R, a1, c1
  
  def define_distributions1(self, k1, log_k2):
    a2 = (self.sigma1*self.eps) / self.sigma0**2
    c2 = -1/2 * (1 - self.sigma1**2 / self.sigma0**2)
    U = lognorm(np.abs(a2), loc=0, scale=exp(self.b(k1)))
    if log_k2 == 0:
      return U, None, a2, c2
    # self._debug_print(f'scale: {self.c2*sqrt(2*self.dim)}')
    # self._debug_print(f'mu: {self.c2 * self.dim + log_k2 - log(k1)}')
    R = lognorm(np.abs(c2*sqrt(2*self.dim)), loc=0, scale=exp(c2 * self.dim + log_k2 - log(k1)))
    return U, R, a2, c2
  
  def define_distributions_certificate(self, k1, log_k2):
    ratio = 1/(self.sigma0**2) - 1/(self.sigma1**2)
    a1 = (self.sigma0 * self.eps) / (self.sigma0**2)
    b1 = -self.eps**2 / (2*self.sigma0**2) + log(k1)
    a2 = 1/2 * self.sigma0**2 * ratio * sqrt(2*self.dim)
    b2 = -self.eps**2/(2*self.sigma1**2) + log_k2 + 1/2 * self.sigma0**2 * ratio * self.dim
    a3 = 1/2 * self.sigma0**2 * ratio
    a4 = (-self.sigma0*self.eps)/(self.sigma1**2)
    U = lognorm(np.abs(a1), 0, 1)
    R = lognorm(np.abs(a2), 0, exp(b2))
    return U, R, b1, a2, a3, a4

  def compute_integral(self, U, R, a, c, k1, log_k2):
    if log_k2 == 0: # => cohen certificate
      result = U.cdf(1)
    else:
      integrand = Integrand(U, R, a, c)
      integ = vegas.Integrator([[0, 1], [0, 1]])
      integ(integrand, nitn=self.nitn_train, neval=self.neval_train)
      result = integ(integrand, nitn=self.nitn, neval=self.neval).mean
    return result
  
  def compute_certificate(self, k1, log_k2):
    if log_k2 == 0: # => cohen certificate
      a1 = (self.sigma0 * self.eps) / (self.sigma0**2)
      b1 = - self.eps**2 / (2*self.sigma0**2) + log(k1)
      U = lognorm(a1, 0, exp(b1))
      result = 1 - U.cdf(1)
    else:
      U, R, b1, a2, a3, a4 = self.define_distributions_certificate(k1, log_k2)
      integrand_certificate = IntegrandCertificate(U, R, b1, a2, a3, a4)
      integ = vegas.Integrator([[0, 1], [0, 1]])
      integ(integrand_certificate, nitn=self.nitn_train, neval=self.neval_train)
      result = integ(integrand_certificate, nitn=self.nitn, neval=self.neval)
    return result
    
  def compute_p0(self, *args):
    U, R, a, c = self.define_distributions0(*args)
    return self.compute_integral(U, R, a, c, *args)

  def compute_p1(self, *args):
    U, R, a, c = self.define_distributions1(*args)
    return self.compute_integral(U, R, a, c, *args)
 
  def compute_p0_p1(self, *args):
    return self.compute_p0(*args), self.compute_p1(*args)

  def adjust_k1(self, k1, log_k2):
    start = 0.
    end = k1 + 0.1
    self._debug_print('\n-- Adjust k1 --')
    while True:
      k1 = (start + end) / 2
      U, R, a, c = self.define_distributions0(k1, log_k2)
      p0 = self.compute_integral(U, R, a, c, k1, log_k2)
      self._debug_print(f'start {start:.5f}, end {end:.5f}, p0 {p0:.5f} / {self.pABar_sigma0}')
      if self.is_close_bellow(p0, self.pABar_sigma0):
        break
      if p0 < self.pABar_sigma0:
        start = k1
      else:
        end = k1
    return k1, log_k2, p0
  
  def adjust_k2(self, k1, log_k2):
    c2 = -1/2 * (1 - self.sigma1**2 / self.sigma0**2)
    start = 0.
    end = np.sign(self.sigma0 - self.sigma1) * (c2 * self.dim + 50)
    self._debug_print('\n-- Adjust k2 --')
    while True:
      log_k2 = (start + end) / 2
      U, R, a, c = self.define_distributions1(k1, log_k2)
      p1 = self.compute_integral(U, R, a, c, k1, log_k2)
      self._debug_print(f'start {start:.5f}, end {end:.5f}, p1 {p1:.5f} / {self.pABar_sigma1}')
      if self.is_close_bellow(p1, self.pABar_sigma1):
        break
      if p1 > self.pABar_sigma1:
      # if p1 < self.pABar_sigma1:
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
    self._debug_print(f'==> certificate: {cert.mean:.5f} \u00B1 {cert.sdev:.5f}, Q={cert.Q:.4f}')
    return cert.mean >= 0.5

  def best_certificate(self):
    if not self.check_better_certificate():
      return self.cert_cohen
    start = self.cert_cohen - 0.01
    end = start + 0.15
    n_iter = 0
    while (end - start) >= self.bs_cert_tol:
      self._debug_print(f'\n--- ITERATION {n_iter} ---')
      self._debug_print(f'start: {start:.4f}, end: {end:.4f}')
      eps = (start + end) / 2
      if self.check_certificate(eps):
        start = eps
      else:
        end = eps
      n_iter += 1
    return start


def find_best_sigma(cert, best_cert, sigma):
  start = sigma
  end = sigma + 0.3
  while (end - start) >= 0.001:
    sigma = (start + end) / 2
    cert.sigma1 = sigma
    current_cert = cert.best_certificate()
    if current_cert < best_cert:
      start = sigma
    else:
      end = sigma
  return start



if __name__ == '__main__':

  parser = argparse.ArgumentParser(
      description='Compute Randomized Smoothing Certificate.')
  parser.add_argument("--dim", type=int, help="Dimension.")
  args = parser.parse_args()

  # We need to init cuda before loading MPI
  # in order to avoid a bug with mpi4py
  # torch.cuda.init()

  # ignore numpy warnings (maybe deal with it)
  np.seterr(all="ignore")

  from mpi4py import MPI
  dist = MPI.COMM_WORLD
  size = dist.Get_size()
  rank = dist.Get_rank()

  sigma0 = 0.50
  sigma1 = 0.55
  pABar_sigma0 = 0.80
  pABar_sigma1 = 0.85
  dims = [60, 80, 100, 120, 150, 200, 250]
  nitn = 30
  neval = 5000000 / size
  verbose = False

  best_certificates = {}
  cert = Certificate(dist, sigma0, sigma1, pABar_sigma0, pABar_sigma1, dims[0],
                     nitn, neval, nitn_train=nitn, neval_train=neval, verbose=verbose)
  start = time.time()
  best_cert = cert.best_certificate()
  if rank == 0:
    print(f'sigma0 {sigma0}, sigma1 {sigma1}')
    print(f'Dim {cert.dim}, best certificate is {best_cert}, duration: {time.time() - start:.3f}')

  for dim in dims:
    cert.dim = dim
    start = time.time()
    sigma1, best_cert = find_best_sigma(cert, best_cert, sigma1)
    if rank == 0:
      print(f'sigma0 {sigma0}, sigma1 {sigma1}')
      print(f'Dim {cert.dim}, best certificate is {best_cert}, duration: {time.time() - start:.3f}')



