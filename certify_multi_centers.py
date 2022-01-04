import numpy as np
from numpy import exp, pi
from scipy.stats import chi, norm

import argparse
import time
import numpy as np
import vegas
from numpy import pi, exp, log, sqrt
from scipy.stats import norm #, lognorm, halfnorm, expon

# from scipy.special import gamma


class Integrand(vegas.BatchIntegrand):

  def __init__(self, dist, eps, di, k_values, sigma):
    self.dist = dist
    self.k_values = k_values
    self.eps = eps
    self.sigma = sigma
    self.di = di

  def indicatrice(self, x):

    g = lambda x: exp(-x / (2*self.sigma**2))
    left_term = g(self.eps**2 - 2*x[:, 0]*self.eps)

    S = np.zeros_like(left_term)
    if len(self.k_values) > 1:
      for i, k in enumerate(self.k_values[1:], 1):
        # S += k * (g((x[:, i] - self.di)**2) / g(x[:, i]**2))
        # S += k * (g( x[:, i]**2 - 2*x[:, i]*self.di + self.di**2) / g(x[:, i]**2))
        # S += k * (g( x[:, i]**2) * g(- 2*x[:, i]*self.di) * g(self.di**2)) / g(x[:, i]**2))
        S += k * g(self.di**2 - 2*x[:, i]*self.di)

    ind = (left_term - S <= self.k_values[0]) * 1.
    return ind

  def __call__(self, x):
    pdfs = [dist.pdf(x[:, i]) for i, dist in enumerate(self.dist)]
    return np.product(pdfs, axis=0) * self.indicatrice(x)


class Certificate:
  """ Compute certificate from generalized Neyman-Pearson Lemma """

  def __init__(self, pABar, n_centers, di, sigma, nitn, neval, verbose=False):

    self.sigma = sigma
    self.n_centers = n_centers
    self.pABar = np.array(pABar)
    self.di = di
    assert self.n_centers == len(pABar)

    self.nitn = nitn
    self.neval = neval
    self.verbose = verbose

    self.adjust_k_values = [0, 5]
    self.adjust_n_iter = 20

    # binary search parameters
    self.bs_cert_tol = 0.001
    self.bs_k_values_close_tol = 0.002

    # Values from Cohen et al.
    self.cert_cohen = self.sigma * norm.ppf(self.pABar[0])
    self.cert_cohen = max(0, self.cert_cohen)
    k0 = self.k_cohen(self.cert_cohen)
    self.debug_print(f'Certificat Cohen et al.: {self.cert_cohen}')
    self.debug_print(f'k0 cohen: {k0}')

    k_values = [k0] + [0] * (self.n_centers-1)
    cert = self.check_certificate(self.cert_cohen, k_values)


  def is_close_bellow(self, x, y):
    if type(x) == np.ndarray:
      return all((y - self.bs_k_values_close_tol <= x) & (x <= y))
    return (y - self.bs_k_values_close_tol <= x) & (x <= y)

  def debug_print(self, msg):
    if self.verbose:
      print(msg)

  def k_cohen(self, eps):
    return exp((eps / self.sigma * norm.ppf(self.pABar[0])) - eps**2 / (2 * self.sigma**2))

  def _integral(self, dist, eps, k_values):
    n_iter = 0
    nitn, neval = self.nitn, self.neval
    while True:
      integrand = Integrand(dist, eps, self.di, k_values, self.sigma)
      integ = vegas.Integrator([[-20, 20]]*self.n_centers)
      integ(integrand, nitn=nitn, neval=neval, alpha=0.1)
      result = integ(integrand, nitn=nitn, neval=neval, alpha=0.1, adapt=False)
      # print(result.summary())
      # print(result.mean, result.Q, nitn, neval)
      if result.Q > 0.6:
        break
      if n_iter > 10:
        neval += 20000
        n_iter = 0
      n_iter += 1
    return result

  def compute_integral(self, eps, k_values, p):
    dist = [norm(0, self.sigma)]
    for i in range(1, self.n_centers):
      dist.append(norm(self.di if p == i else 0, self.sigma))
    return self._integral(dist, eps, k_values).mean

  def compute_certificate(self, eps, k_values):
    dist = [norm(eps, self.sigma)]
    for i in range(1, self.n_centers):
      dist.append(norm(0, self.sigma))
    return self._integral(dist, eps, k_values)

  def adjust_k(self, eps, k_values, k_index):
    start, end = self.adjust_k_values
    self.debug_print(f'\n-- Adjust k{k_index} --')
    while True:
      k = (start + end) / 2
      k_values[k_index] = k
      p = self.compute_integral(eps, k_values, k_index)
      pABar = self.pABar[k_index]
      self.debug_print(
        f'start {start:3.5f}, end {end:3.5f}, p{k_index} {p:.5f} / {pABar}, k{k_index} = {k:.4f}')
      if self.is_close_bellow(p, self.pABar[k_index]):
        break
      if p < self.pABar[k_index]:
        start = k
      else:
        end = k
    k_values[k_index] = k
    return k_values

  def find_k_values(self, eps):
    k_values = np.array([0.0] * self.n_centers)
    k_values[0] = self.k_cohen(eps) - 0.02
    probas = np.array([0.] * self.n_centers)
    break_while = False
    while True:
      for k_index in range(self.n_centers)[::-1]:
        k_values = self.adjust_k(eps, k_values, k_index)
        for i in range(self.n_centers):
          probas[i] = self.compute_integral(eps, k_values, k_index)
        probas_str = ', '.join([f'p{i}: {p1:.3f}/{p2:.3f}' for i, (p1, p2) in enumerate(zip(probas, self.pABar))])
        self.debug_print(probas_str)
        if self.is_close_bellow(probas, self.pABar):
          break_while = True
          break
      if break_while:
        break
    return k_values

  def check_certificate(self, eps, k_values=None):
    if not k_values:
      k_values = self.find_k_values(eps)
    cert = self.compute_certificate(eps, k_values)
    self.debug_print('\n-- Computing Certificate -- ')
    self.debug_print(', '.join([f'k{i} = {v}' for i, v in enumerate(k_values)]))
    cert1 = cert.mean - cert.sdev
    cert2 = cert.mean + cert.sdev
    # self.debug_print(f'Certificate for {eps:.5f}: {cert.mean:.5f} \u00B1 {cert.sdev:.5f}, Q={cert.Q:.4f}')
    self.debug_print(f'Certificate for {eps:.5f}: {cert1:.5f}/{cert2:.5f}, Q={cert.Q:.4f}')
    return cert.mean >= 0.5

  def best_certificate(self):
    start = max(0, self.cert_cohen - 0.02)
    end = self.cert_cohen + 0.05
    n_iter = 1
    while (end - start) >= self.bs_cert_tol:
      self.debug_print(f'\n--- ITERATION {n_iter} ---')
      self.debug_print(f'start: {start:.4f}, end: {end:.4f}')
      eps = (start + end) / 2
      if self.check_certificate(eps):
        start = eps
      else:
        end = eps
      n_iter += 1
    return start

if __name__ == '__main__':

  sigma = 0.25
  # pABar = [0.65, 0.65, 0.65]
  pABar = [0.65, 0.65]
  # pABar = [0.65]
  n_centers = len(pABar)
  di = 0.25

  nitn = 10
  neval = 20000
  verbose = True

  cert = Certificate(pABar, n_centers, di, sigma, nitn, neval, verbose)
  cohen = cert.cert_cohen

  # eps = cohen + 0.002
  # print(f'Try for {eps}')
  # print(cert.check_certificate(eps))

  start = time.time()
  best_cert = cert.best_certificate()
  print(f'Best certificate is {best_cert}, duration: {time.time() - start:.3f}')
  print(f'Certificat Cohen et al.: {cert.cert_cohen}')



