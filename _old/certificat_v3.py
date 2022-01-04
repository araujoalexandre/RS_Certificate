
import time
import torch
import numpy as np
import vegas
from multiprocessing.pool import ThreadPool
from numpy import exp, log, sqrt
from scipy.stats import norm
from scipy.special import erfinv
from torch.distributions.log_normal import LogNormal


class Integral(vegas.BatchIntegrand):

  def __init__(self, U, R, a1, c1, device):
    self.U = U
    self.R = R
    self.a1 = a1
    self.c1 = c1
    self.device = f'cuda:{device}'

  def __call__(self, x):
    x = torch.Tensor(x).to(self.device)
    x, y = x[:, 0], x[:, 1]
    x_pdf = torch.exp(self.U.log_prob(x))
    y_pdf = torch.exp(self.R.log_prob(y))
    res = x_pdf * y_pdf * ((x - y * x**(2*self.c1/self.a1) <= 1.)*1.)
    return np.double(res.cpu().numpy())


class IntegralCertificate(vegas.BatchIntegrand):

  def __init__(self, U, R, b1, a2, a3, a4):
    self.U = U
    self.R = R
    self.b1 = b1
    self.a2 = a2
    self.a3 = a3
    self.a4 = a4

  def __call__(self, x):
    x = torch.Tensor(x).cuda()
    x, y = x[:, 0], x[:, 1]
    x_pdf = torch.exp(self.U.log_prob(x))
    y_pdf = torch.exp(self.R.log_prob(y))
    res = x_pdf * y_pdf * (x * exp(self.b1) + y * x**(2*self.a3/self.a2) * x**(self.a4/self.a2) >= 1.)
    return np.double(res.cpu().numpy())




class Certificate:
  """ Compute certificate from generalized Neyman-Pearson Lemma"""
  
  def __init__(self, sigma0, sigma1, pABar_sigma0, pABar_sigma1, dim, nitn, neval):
    
    self.sigma0 = sigma0
    self.sigma1 = sigma1
    self.pABar_sigma0 = pABar_sigma0
    self.pABar_sigma1 = pABar_sigma1
    
    self.dim = dim
    
    self.nitn = nitn
    self.neval = neval

    self.bs_cert_tol = 0.001
    self.bs_k_values_close_tol = 0.005
    self.is_close_bellow = lambda x, y: y - self.bs_k_values_close_tol <= x <= y

    self.percentile = 1 - 1e-16
    
    # Values from Cohen et al.
    self.cert_cohen = self.sigma0 * norm.ppf(self.pABar_sigma0)
    print('certificat cohen', self.cert_cohen)
    
  def compute_k1_cohen(self, eps):
    """ Compute the constant k1 in the context of Cohen certificate """
    return exp((eps / self.sigma0 * norm.ppf(self.pABar_sigma0)) - self.eps**2 / (2 * self.sigma0**2))
    
  def check_better_certificate(self):
    """ Check if a better certificate is possible """
    p0, p1 = self.compute_p0_p1(self.k1_cohen, 0, 0)
    if p1 <= pABar_sigma1:
      print('Good News, a better certificate is possible !')
      return True
    else:
      print('Nop ! We cannot do better than Cohen et al.')
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

  # @vegas.bacompute_integraltchintegrand
  # def batch_integrand(self, x):
  #   x = torch.Tensor(x).cuda()
  #   x, y = x[:, 0], x[:, 1]
  #   x_pdf = torch.exp(self.U.log_prob(x))
  #   y_pdf = torch.exp(self.R.log_prob(y))
  #   res = x_pdf * y_pdf * ((x - y * x**(2*self.c1/self.a1) <= 1.)*1.)
  #   return np.double(res.cpu().numpy())

  # @vegas.batchintegrand
  # def batch_integrand_certificate(self, x):
  #   x = torch.Tensor(x).cuda()
  #   x, y = x[:, 0], x[:, 1]
  #   x_pdf = torch.exp(self.U.log_prob(x))
  #   y_pdf = torch.exp(self.R.log_prob(y))
  #   res = x_pdf * y_pdf * (x * exp(self.b1) + y * x**(2*self.a3/self.a2) * x**(self.a4/self.a2) >= 1.)
  #   return np.double(res.cpu().numpy())

  def compute_integral(self, U, R, k1, log_k2, support, device):
    if log_k2 == 0: # equiv k2 = 0 => cohen certificate
      result = U.cdf(torch.Tensor([1.]))
    else:
      m1, m2 = support
      batch_integrand = Integral(U, R, self.a1, self.c1, device) 
      integ = vegas.Integrator([[0, m1], [0, m2]])
      integ(batch_integrand, nitn=self.nitn, neval=self.neval)
      result = integ(batch_integrand, nitn=self.nitn, neval=self.neval).mean
    return result
  
  def compute_certificate(self, k1, log_k2):
    if log_k2 == 0: # equiv k2 = 0 => cohen certificate
      a1 = (self.sigma0 * self.eps) / (self.sigma0**2)
      b1 = - self.eps**2 / (2*self.sigma0**2) + log(k1)
      U = LogNormal(b1, a1)
      result = 1 - U.cdf(torch.Tensor([1.]))
    else:
      U, R, m1, m2 = self.define_distributions_certificate(k1, log_k2)
      batch_integrand_certificate = IntegralCertificate(U, R, self.b1, self.a2, self.a3, self.a4)
      integ = vegas.Integrator([[0, m1], [0, m2]])
      integ(batch_integrand_certificate, nitn=self.nitn, neval=self.neval)
      result = integ(batch_integrand_certificate, nitn=self.nitn, neval=self.neval).mean
    return result
  
  def compute_p0_p1(self, *args):
    return self.compute_p0(args), self.compute_p1(args)
  
  def compute_p0(self, args):
    # print('compute_p0', args)
    k1, log_k2, device = args
    U, R, *support = self.define_distributions0(k1, log_k2)
    return self.compute_integral(U, R, k1, log_k2, support, device)

  def compute_p1(self, args):
    # print('compute_p1', args)
    k1, log_k2, device = args
    U, R, *support = self.define_distributions1(k1, log_k2)
    return self.compute_integral(U, R, k1, log_k2, support, device)

  def distributed_compute_p0(self, k1_values, log_k2):
    pool = ThreadPool(processes=7)
    devices = np.arange(7)
    args = [ (k1, log_k2, device) for k1, device in zip(k1_values, devices) ]
    results = pool.map(self.compute_p0, args)
    pool.close()
    return np.array([1] + results + [1])

  def distributed_compute_p1(self, k1, log_k2_values):
    pool = ThreadPool(processes=7)
    devices = np.arange(7)
    args = [ (k1, log_k2, device) for log_k2, device in zip(log_k2_values, devices) ]
    results = pool.map(self.compute_p1, args, 1)
    pool.close()
    return np.array([1] + results + [1])

  def adjust_k1(self, k1, log_k2):
    is_close_bellow = self.is_close_bellow
    pABar_sigma0 = self.pABar_sigma0
    start = 0
    end = k1 + 0.1
    while True:
      k1_values = np.linspace(start, end, num=7+2)
      results = self.distributed_compute_p0(k1_values[1:-1], log_k2)
      for ix in range(len(results) - 1):
        if results[ix] < pABar_sigma0 < results[ix+1]:
          break
      start, end = k1_values[ix], k1_values[ix+1]
      if is_close_bellow(results[ix], pABar_sigma0):
        k1, p0 = k1_values[ix], results[ix]
        break
      if is_close_bellow(results[ix+1], pABar_sigma0):
        k1, p0 = k1_values[ix+1], results[ix+1]
        break
    return k1, log_k2, p0
  
  def adjust_k2(self, k1, log_k2):
    is_close_bellow = self.is_close_bellow
    pABar_sigma1 = self.pABar_sigma1
    start = 0.
    end = -(self.c2 * self.dim + 50)
    while True:
      log_k2_values = np.linspace(start, end, num=7+2)
      results = self.distributed_compute_p1(k1, log_k2_values[1:-1])
      for ix in range(len(results) - 1):
        if results[ix] > pABar_sigma1 > results[ix+1]:
          break
      start, end = log_k2_values[ix], log_k2_values[ix+1]
      if is_close_bellow(results[ix], pABar_sigma1):
        log_k2, p1 = log_k2_values[ix], results[ix]
        break
      if is_close_bellow(results[ix+1], pABar_sigma1):
        log_k2, p1 = log_k2_values[ix+1], results[ix+1]
        break
    return k1, log_k2, p1
  
  def find_k_values(self):
    k1 = self.compute_k1_cohen(self.eps)
    log_k2 = 0
    p0, p1 = self.compute_p0_p1(k1, log_k2, 0)
    while True:
      k1, log_k2, p1 = self.adjust_k2(k1, log_k2)
      p0 = self.compute_p0([k1, log_k2, 0])
      if self.is_close_bellow(p0, self.pABar_sigma0) and self.is_close_bellow(p1, self.pABar_sigma1):
        break
      k1, log_k2, p0 = self.adjust_k1(k1, log_k2)
      p1 = self.compute_p1([k1, log_k2, 0])
      if self.is_close_bellow(p0, self.pABar_sigma0) and self.is_close_bellow(p1, self.pABar_sigma1):
        break
    return k1, log_k2

  def check_certificate(self, eps):
    self.eps = eps
    time_start = time.time()
    k1, log_k2 = self.find_k_values()
    print('time to find k1, k2: {}'.format(time.time() - time_start))
    cert = self.compute_certificate(k1, log_k2)
    return cert >= 0.5

  def best_certificate(self):
    start = self.cert_cohen - 0.01
    end = start + 0.15
    while (end - start) >= self.bs_cert_tol:
      print(f'start: {start:.4f}, end: {end:.4f}')
      eps = (start + end) / 2
      if self.check_certificate(eps):
        start = eps
      else:
        end = eps
    return start


if __name__ == '__main__':

  a = torch.randn(20, 20).cuda()
  sigma0 = 0.50
  sigma1 = 0.55
  pABar_sigma0 = 0.80
  pABar_sigma1 = 0.85
  dim = 100
  nitn = 60
  neval = 3000000
  cert = Certificate(sigma0, sigma1, pABar_sigma0, pABar_sigma1, dim-2, nitn, neval)
  print(cert.best_certificate())




