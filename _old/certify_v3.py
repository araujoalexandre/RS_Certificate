
import argparse
import time
import torch
import numpy as np
import vegas
from numpy import pi, exp, log, sqrt
from scipy.stats import norm, lognorm, halfnorm, expon

"""
Tentative certificate with sampling from the norm and the direction separatly
"""


class Integrand(vegas.BatchIntegrand):

  def __init__(self, f, f_s0, f_s1, dim, eps, k1, k2):
    self.f = f
    self.f_s0 = f_s0
    self.f_s1 = f_s1
    self.eps = eps
    self.dim = dim
    self.k1 = k1
    self.k2 = k2

  def _left_term(self, r, z):
    z = z - self.eps
    self.f_s0(sqrt(r**2 + z**2)) / (r**2 + z**2) * (r / sqrt(r**2 + z**2))**self.dim

  def indicatrice(self, r, z):
    left_term = self._left_term(r, z)
    k1_term = self.k1 * self.f_s0(sqrt(r**2 + z**2)) / (r**2 + z**2) * (r / sqrt(r**2 + z**2))
    k2_term = self.k2 * self.f_s1(sqrt(r**2 + z**2)) / (r**2 + z**2) * (r / sqrt(r**2 + z**2))
    return ( left_term <= (k1_term + k2_term) )*1.

  def __call__(self, x):
    r, z = x[:, 0], x[:, 1]
    indicatrice = self.indicatrice(r, z)
    return self.f.pdf(sqrt(r**2 + z**2)) / (r**2 + z**2) * (r / sqrt(r**2 + z**2))**self.dim * indicatrice



class IntegrandCertificate(vegas.BatchIntegrand):

  def __init__(self, f_s0, f_s1, eps, k1, k2):
    self.f_s0 = f_s0
    self.f_s1 = f_s1
    self.eps = eps
    self.k1 = k1
    self.k2 = k2

  def indicatrice(self, r, z):
    left_term = self.f_s0.pdf(sqrt(r**2 + (z - self.eps)**2)) / sqrt(r**2 + (z - self.eps)**2)
    k1_term = self.k1 * self.f_s0.pdf(sqrt(r**2 + z**2))
    k2_term = self.k2 * self.f_s1.pdf(sqrt(r**2 + z**2))
    return ( left_term <= 1/sqrt(r**2 + z**2) * (k1_term + k2_term) )*1.

  def __call__(self, x):
    r, z = x[:, 0], x[:, 1]
    indicatrice = self.indicatrice(r, z)
    return  1/pi * 1/sqrt(r**2 + (z - self.eps)**2) * self.f_s0.pdf(sqrt(r**2 + (z - self.eps)**2)) * indicatrice



class Certificate:
  """ Compute certificate from generalized Neyman-Pearson Lemma"""

  def __init__(self, dist, sigma0, sigma1, pABar_sigma0, pABar_sigma1,
               nitn, neval, nitn_train=None, neval_train=None, verbose=False):

    self.dist = dist
    self.rank = dist.Get_rank()
    self.verbose = verbose

    self.sigma0 = sigma0
    self.sigma1 = sigma1
    self.pABar_sigma0 = pABar_sigma0
    self.pABar_sigma1 = pABar_sigma1

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
    if self.verbose and self.rank == 0:
      print(msg)

  def compute_integral(self, k1, k2, p):
    f_s0 = expon(0, self.sigma0)
    f_s1 = expon(0, self.sigma1)
    if p == 'p0':
      integrand = Integrand(f_s0, f_s0, f_s1, self.eps, k1, k2)
    elif p == 'p1':
      integrand = Integrand(f_s1, f_s0, f_s1, self.eps, k1, k2)

    integ = vegas.Integrator([[0, 10], [-10, 10]])
    integ(integrand, nitn=self.nitn_train, neval=self.neval_train)
    result = integ(integrand, nitn=self.nitn, neval=self.neval).mean
    return result

  def compute_certificate(self, k1, k2):
    f_s0 = expon(0, self.sigma0)
    f_s1 = expon(0, self.sigma1)
    integrand_certificate = IntegrandCertificate(f_s0, f_s1, self.eps, k1, k2)
    integ = vegas.Integrator([[0, 10], [-10, 10]])
    integ(integrand_certificate, nitn=self.nitn_train, neval=self.neval_train)
    result = integ(integrand_certificate, nitn=self.nitn, neval=self.neval)
    return result

  def compute_p0(self, k1, k2):
    return self.compute_integral(k1, k2, 'p0')

  def compute_p1(self, k1, k2):
    return self.compute_integral(k1, k2, 'p1')

  def compute_p0_p1(self, *args):
    return self.compute_p0(*args), self.compute_p1(*args)

  def adjust_k1(self, k1, k2):
    start = 0.
    end = 50
    self._debug_print('\n-- Adjust k1 --')
    while True:
      k1 = (start + end) / 2
      p0 = self.compute_p0(k1, k2)
      self._debug_print(f'start {start:3.5f}, end {end:3.5f}, p0 {p0:.5f} / {self.pABar_sigma0}, k1 = {k1:.4f} ')
      if self.is_close_bellow(p0, self.pABar_sigma0):
        break
      if p0 < self.pABar_sigma0:
        start = k1
      else:
        end = k1
    return k1, k2, p0

  def adjust_k2(self, k1, k2):
    start = 0.
    end = 50
    self._debug_print('\n-- Adjust k2 --')
    while True:
      k2 = (start + end) / 2
      p1 = self.compute_p1(k1, k2)
      self._debug_print(f'start {start:3.5f}, end {end:3.5f}, p1 {p1:.5f} / {self.pABar_sigma1}, k2 = {k2:.4f} ')
      if self.is_close_bellow(p1, self.pABar_sigma1):
        break
      if p1 < self.pABar_sigma1:
        start = k2
      else:
        end = k2
    return k1, k2, p1

  def find_k_values(self):
    k1, k2 = 0.1, 0.1
    p0, p1 = 0., 0.
    while True:
      k1, k2, p0 = self.adjust_k1(k1, k2)
      p1 = self.compute_p1(k1, k2)
      if self.is_close_bellow(p0, self.pABar_sigma0) and self.is_close_bellow(p1, self.pABar_sigma1):
        break
      k1, k2, p1 = self.adjust_k2(k1, k2)
      p0 = self.compute_p0(k1, k2)
      if self.is_close_bellow(p0, self.pABar_sigma0) and self.is_close_bellow(p1, self.pABar_sigma1):
        break
    return k1, k2

  def check_certificate(self, eps):
    self.eps = eps
    k1, k2 = self.find_k_values()
    cert = self.compute_certificate(k1, k2)
    self._debug_print('\n-- Computing Certificate -- ')
    self._debug_print(f'k1 = {k1:.6f}, k2 = {k2:.6f}')
    self._debug_print(f'Certificate: {cert.mean:.5f} \u00B1 {cert.sdev:.5f}, Q={cert.Q:.4f}')
    return cert.mean >= 0.5

  def best_certificate(self):
    start = 0.
    end = self.cert_cohen + 0.15
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




if __name__ == '__main__':

  parser = argparse.ArgumentParser(
      description='Compute Randomized Smoothing Certificate.')
  parser.add_argument("--sigma0", type=float, help="Sigma0.")
  parser.add_argument("--sigma1", type=float, help="Sigma1.")
  parser.add_argument("--pABar_sigma0", type=float, help="pABar_sigma0.")
  parser.add_argument("--pABar_sigma1", type=float, help="pABar_sigma1.")
  args = parser.parse_args()

  # We need to init cuda before loading MPI
  # in order to avoid a bug with mpi4py
  torch.cuda.init()

  # ignore numpy warnings (maybe deal with it)
  np.seterr(all="ignore")

  from mpi4py import MPI
  dist = MPI.COMM_WORLD
  size = dist.Get_size()
  rank = dist.Get_rank()

  sigma0 = args.sigma0
  sigma1 = args.sigma1
  pABar_sigma0 = args.pABar_sigma0
  pABar_sigma1 = args.pABar_sigma1
  nitn = 20
  neval = 1000000 / size
  verbose = True

  if rank == 0:
    print('-- Context --')
    print(f'sigma0 {sigma0}, sigma1 {sigma1}')
    print(f'pABar_sigma0 {pABar_sigma0}, pABar_sigma1 {pABar_sigma1}')

  cert = Certificate(dist, sigma0, sigma1, pABar_sigma0, pABar_sigma1,
                     nitn, neval, nitn_train=nitn, neval_train=neval, verbose=verbose)

  start = time.time()
  best_cert = cert.best_certificate()
  if rank == 0:
    print(f'Best certificate is {best_cert}, duration: {time.time() - start:.3f}')

  # start_global = time.time()
  # for eps in np.arange(0.25, 3.25, 0.25):
  #   start = time.time()
  #   certificate = cert.check_certificate(eps)
  #   if rank == 0:
  #     print(f'Certification at {eps}: {certificate}, duration {time.time() - start:.3f}')
  # print(f'global duration: {time.time() - start_global:.3f}')



