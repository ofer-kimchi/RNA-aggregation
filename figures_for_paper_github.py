#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time
import copy             
from sklearn.linear_model import LinearRegression
from scipy.special import gamma, polygamma
import pickle
import os
from scipy.sparse.csgraph import connected_components
from scipy.sparse import dok_matrix
from scipy.optimize import fsolve, root
from mpmath import lerchphi, polylog

# =============================================================================
# =============================================================================
# =============================================================================
# # # Function definitions
# =============================================================================
# =============================================================================
# =============================================================================


def save(obj, filename):
# =============================================================================
#     Save an object to a file
# =============================================================================
    if not filename[-7:] == '.pickle':
        filename = filename + '.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=4) #protocol 4 came out with Python version 3.4
    
    
def load(filename):
# =============================================================================
#     Load an object from a file
# =============================================================================
    if not filename[-7:] == '.pickle':
        filename = filename + '.pickle'
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data
 
folder_name_init = '/Users/your_folder_name_here/stored_partition_fxns/'
   
def log_sum_exp(a, boltzmann_offset=None):
    # Given an array a, exponentiate each term, take the sum, and then take the log.
    # Also use an offset so that the exponentials don't get too large

    a = np.array(a)
    if boltzmann_offset is None and any(a > 650):  # np.exp(700) is almost infinity
        boltzmann_offset = 350  # np.median(a)  # takes 0.003 s for array of len 40000. Same time as max or mean (+15%).
    elif boltzmann_offset is None:
        boltzmann_offset = 0
    unnormalized_probs = np.exp(np.array(a) - boltzmann_offset)
    return(np.log(np.sum(unnormalized_probs)) + boltzmann_offset)


def log_subtract_exp(a, b, boltzmann_offset=None):
    # Given two terms a and b, calculate log(exp(a) - exp(b))
    # Also use an offset so that the exponentials don't get too large

    if boltzmann_offset is None and (a > 650 or b > 650):  # np.exp(700) is almost infinity
        boltzmann_offset = np.mean([a, b])
        
    elif boltzmann_offset is None:
        boltzmann_offset = 0
        
    offset_a = a - boltzmann_offset
    offset_b = b - boltzmann_offset
    # e^x - e^y = e^a * (e^(x-a) - e^(y-a))
    return(np.log(np.exp(offset_a) - np.exp(offset_b)) + boltzmann_offset)    
    

all_subdivisions = {}
all_subdivisions[0] = [""]
all_subdivisions[1] = ["()"]

def count_all_subdivisions(m):
    # Construct all combinations of m open and m closed parentheses, and for each,
    # determine the combination of monomers, dimers, trimers, etc. they correspond
    # to, where each strand is comprised of 2 symbols.
    # For example, (())() is a dimer followed by a monomer. 
    # This procedure already ensures there are no pseudoknots
    
    # Code taken from https://stackoverflow.com/questions/4313921/recurrence-approach-how-can-we-generate-all-possibilities-on-braces
    if m in all_subdivisions:
        return all_subdivisions[m]
    
    all_subdivisions[m] = []
    
    for i in np.arange(1,m+1):
        between = count_all_subdivisions(i-1)
        after   = count_all_subdivisions(m-i)
        for b in between:
            for a in after:
                all_subdivisions[m].append("("+b+")"+a)
    
    return all_subdivisions[m]

def count_clusters(parens):
    n_nodes = int(len(parens)/2)
    graph = dok_matrix((n_nodes,n_nodes), dtype=int)
    labels = [i//2 for i in range(len(parens))]
    while parens != '':
        ind = parens.find('()')
        graph[labels[ind], labels[ind + 1]] = 1
        parens = parens.replace('()', '', 1)
        del labels[ind:ind+2]
      
    n_components, ls = connected_components(csgraph=graph.tocsr(), directed=False, return_labels=True)
    _, clusters = np.unique(ls, return_counts=True)
    return clusters


def ways_to_connect_m(m):
  # Returns the ways to connect m-mers and the multiplicity of each way
    if m > 15:  # too large -- count all subdivisions will have issues
        return(False)
    
    try:
        all_cluster_possibilities = load(
            folder_name_init + 'ways_to_connect_m/' + 
            'all_cluster_possibilities_' + str(m))
        all_cluster_counts = load(
            folder_name_init + 'ways_to_connect_m/' + 
            'all_cluster_counts_' + str(m))
        return(all_cluster_possibilities, all_cluster_counts)
    except:
        subs = count_all_subdivisions(m)
        all_cluster_possibilities = []
        all_cluster_counts = []
        for s in subs:
            cluster = sorted(count_clusters(s))
            if len(cluster) != 1:  # don't include [m] in the result -- we want only subdivisions
                if cluster in all_cluster_possibilities:
                    idx = all_cluster_possibilities.index(cluster)
                    all_cluster_counts[idx] += 1
                else:
                    all_cluster_possibilities += [cluster]
                    all_cluster_counts += [1]
                               
        save(all_cluster_possibilities, folder_name_init + 'ways_to_connect_m/' + 
             'all_cluster_possibilities_' + str(m))
        save(all_cluster_counts, folder_name_init + 'ways_to_connect_m/' + 
             'all_cluster_counts_' + str(m))
        return(all_cluster_possibilities, all_cluster_counts)


def create_complexes_from_multimers(multimers_stored_log_boltzmann_factor, ns_to_compute,
                                    DNA=False, T=273.15+37):
    if DNA:
        E_cost = 1.96
    else:
        E_cost = 4.09
    translational_free_energy_cost_of_binding = E_cost - kB * T * np.log(rhoH20)


    def create_complexes_from_multimers_general(ns_to_compute):
        # max_m = len(ns_to_compute)
        complexes = np.zeros((max_m, max(ns_to_compute)))
        
        for n in range(1, ns_to_compute[0]):
            complexes[0][n] = multimers_stored_log_boltzmann_factor[(n,)]
            
        for m in range(2, max_m + 1):
            disconnected_possibilities, disconnected_multiplicities = ways_to_connect_m(m)
            for n in range(1 + (m>2), ns_to_compute[m - 1]):
                correction = []
                for possibility, multiplicity in zip(disconnected_possibilities, disconnected_multiplicities):
                    correction += [np.log(multiplicity) + sum(
                        [complexes[i - 1][n] + np.log(i) for i in possibility])]
                complexes[m - 1][n] = (log_subtract_exp(
                    multimers_stored_log_boltzmann_factor[(n,) * m],  
                    log_sum_exp(correction)) - np.log(m))
        return(complexes)
        
    complexes = create_complexes_from_multimers_general(ns_to_compute)
        
    corrected_log_Z = np.zeros((max_m, max(ns_to_compute)))
    for m in range(max_m):
        corrected_log_Z[m, 2:ns_to_compute[m]] = (complexes[m, 2:ns_to_compute[m]] -
                       m * translational_free_energy_cost_of_binding / (kB * T))

    return(complexes, corrected_log_Z)
    

kB = 0.0019872
b = 0.8 / 0.33
vs = 0.02
g = 3 / (2 * b)
logVs = np.log(vs)
rhoH20 = 55.14

def closedNet0EntropyFxn(s1, g):
    ent = (3/2) * np.log(g / (np.pi * s1))
    return(kB * (ent + logVs))
    

def analytical_simp_predicted_logZ_all_bonds(n, m, eff_S_loop, adjacent_stickers_can_bind,
                                             FE_from_one_bp=-10, alpha_0=0.42, alpha_n=0, 
                                             DNA=False, T=273.15+37, include_rhoH2O=True):
    # For monomers, with no loop entropy, allowing neighbor pairings, predicts logZ. 
    # n = # repeats in each strand; m = # strands - 1 (so m=0 is monomers; m=1 is dimers, etc.)
    # Compare to multimers((n,))[0] 
    if DNA:
        E_cost = 1.96
    else:
        E_cost = 4.09
    if include_rhoH2O:
        translational_free_energy_cost_of_binding = E_cost - kB * T * np.log(rhoH20)
    else:
        translational_free_energy_cost_of_binding = E_cost

    loop_ent = closedNet0EntropyFxn(eff_S_loop, g)
    beta_eff_Fb = (FE_from_one_bp - T * loop_ent)/(kB * T)
    ent_trans_diff = -translational_free_energy_cost_of_binding/T - loop_ent

    alpha = alpha_0 + n * alpha_n  #n/50 + 0.25 + beta_eff_Fb / 50 # 0.42
    num_stickers = (m + 1) * n + (1-adjacent_stickers_can_bind) * m * alpha  # since m is # strands - 1

    if adjacent_stickers_can_bind:
        if num_stickers % 2: # i.e. if n*(m+1) is odd
            return((num_stickers - 1) * (np.log(2) - beta_eff_Fb / 2) +  
                      np.log(((num_stickers+1)/(2*num_stickers) * np.sqrt((num_stickers-1)*np.pi/2))**(-1)
                               + num_stickers/3 * np.sqrt((num_stickers-1) / (8*np.pi)) * np.exp(beta_eff_Fb)
                              )
                      - np.log(m+1) # symmetry correction
                      + m * ent_trans_diff/kB)  # correction for m bps having translational entropy cost instead of loop
        else:
            return(num_stickers * (np.log(2) - beta_eff_Fb/2) +
                      np.log(
                          ((2 + num_stickers) * np.sqrt(np.pi * num_stickers / 8))**(-1)
                           + np.sqrt(num_stickers / (8*np.pi)) * np.exp(beta_eff_Fb) # higher order term
                          )
                      - np.log(m+1)
                      + m * ent_trans_diff/kB)
    else:
        if ((m + 1) * n) % 2: # i.e. if n*(m+1) is odd
            return(-beta_eff_Fb * (num_stickers - 1) / 2 + 
                       np.log(1 
                              + (num_stickers + 3) * (num_stickers + 1)**2 * (num_stickers - 1) * np.exp(beta_eff_Fb) / 192
                              )
                       - np.log(m+1) # symmetry correction
                       + m * ent_trans_diff/kB  # correction for m bps having translational entropy cost instead of loop
                       )
        else:
            return(-beta_eff_Fb * (num_stickers / 2 - 1) + 
                   np.log( np.exp(-beta_eff_Fb) * (m % 2) +  # if m is even can also have all stickers bound
                       (num_stickers**2 + 2 * num_stickers) / 8 * (
                       1 + 
                       (num_stickers/2 + 2) * (num_stickers/2 + 1) * (num_stickers/2) * (num_stickers/2 - 1) * np.exp(beta_eff_Fb) / 72
                          ))
                   - np.log(m+1) # symmetry correction
                   + m * ent_trans_diff/kB  # correction for m bps having translational entropy cost instead of loop
                   )



def analytical_simp_predicted_logZ_some_bonds(n, m, eff_S_loop, adjacent_stickers_can_bind,
                                             FE_from_one_bp=-10, alpha_0=0.42, alpha_n=0, 
                                             DNA=False, T=273.15+37, include_saddlepoint=True):
    # For monomers, with no loop entropy, allowing neighbor pairings, predicts logZ. 
    # n = # repeats in each strand; m = # strands - 1 (so m=0 is monomers; m=1 is dimers, etc.)
    # Compare to multimers((n,))[0] 
    # This is the prediction if the energy is not so negative that all bonds are typically satisfied

    # num_stickers = (m + 1) * n + (1-adjacent_stickers_can_bind) * m * 0.42  # since m is # strands - 1
    if DNA:
        E_cost = 1.96
    else:
        E_cost = 4.09
    translational_free_energy_cost_of_binding = E_cost - kB * T * np.log(rhoH20)

    loop_ent = closedNet0EntropyFxn(eff_S_loop, g)
    beta_eff_Fb = (FE_from_one_bp - T * loop_ent)/(kB * T)
    ent_trans_diff = -translational_free_energy_cost_of_binding/T - loop_ent

    alpha = alpha_0 + n * alpha_n  #n/50 + 0.25 + beta_eff_Fb / 50 # 0.42
    num_stickers = (m + 1) * n + (1-adjacent_stickers_can_bind) * m * alpha  # since m is # strands - 1
    
    if adjacent_stickers_can_bind:
        log_alpha = (np.log(1 + 2 * np.exp(-beta_eff_Fb / 2)) / 2 + 
                      2 * np.log(2 + np.exp(beta_eff_Fb / 2)) - 
                      np.log(2 * num_stickers * np.pi) - 
                      np.log(2 + num_stickers + np.exp(beta_eff_Fb / 2)) -
                      ent_trans_diff/kB)
        log_gamma = (n * np.log(1 + 2 * np.exp(-beta_eff_Fb/2)) + 
                      ent_trans_diff/kB)
        
        # As a check: This is the exact same thing
        # log_gamma = 0
        # NpStar = num_stickers / (2 + np.exp(beta_eff_Fb / 2))
        # log_alpha = ((num_stickers + 0.5) * np.log(1 + 2 * np.exp(-beta_eff_Fb / 2)) 
        #               -np.log (2 * np.pi * NpStar * (NpStar + 1))
        #               + m * ent_trans_diff/kB  # correction for m bps having translational entropy cost instead of loop
        #               )
        
        if not include_saddlepoint:
            log_saddlepoint = 0
        else:
            # log_neg_second_deriv = np.log(
            #     2 * polygamma(
            #         1, 1 + num_stickers - 2 * num_stickers / (2 + np.exp(beta_eff_Fb / 2))) + 
            #     polygamma(
            #         1, 1 + num_stickers / (2 + np.exp(beta_eff_Fb / 2))))
            NpStar = num_stickers / (2 + np.exp(beta_eff_Fb / 2))
            log_neg_second_deriv = -2 * np.log(NpStar + 1) + np.log( - (
                1 - 4 * (NpStar + 1)**2 * polygamma(1, 1+num_stickers - 2*NpStar) - 
                2 * (NpStar + 1)**2 * polygamma(1, 1 + NpStar)
                ))
            log_saddlepoint = max(0, (np.log(2 * np.pi) -log_neg_second_deriv) / 2)  # since if it's less than 0, sum is dominated by just main term
            #print((np.log(2 * np.pi) -log_neg_second_deriv) / 2)
        return(log_alpha + (m + 1) * log_gamma - np.log(m + 1) + log_saddlepoint)  # for symmetry reasons; affects beta

    else:
        NpStar = ((num_stickers-1) / 2) * (1 - np.exp(beta_eff_Fb / 4) / (np.sqrt(4 + np.exp(beta_eff_Fb / 2))))
        if not include_saddlepoint:
            log_saddlepoint = 0
        else:
            saddlepoint_lambda = lambda x: 1/x**2 - 2 * polygamma(1, x)
            log_neg_second_deriv = np.log(
                -4 * saddlepoint_lambda(num_stickers - 2*NpStar) 
                +saddlepoint_lambda(num_stickers - NpStar)
                -saddlepoint_lambda(NpStar + 1))
            log_saddlepoint = max(0, (np.log(2 * np.pi) - log_neg_second_deriv) / 2)  # since if it's less than 0, sum is dominated by just main term

        return(
            - beta_eff_Fb * NpStar +
            2 * NpStar * np.log(num_stickers / NpStar - 2)
            -np.log (2 * np.pi * NpStar * (NpStar + 1)) + 
            2 * (num_stickers - NpStar) * np.log(1/2 + num_stickers / (2 * (num_stickers - 2*NpStar)))
            + log_saddlepoint
            - np.log(m+1) # symmetry correction
            + m * ent_trans_diff/kB  # correction for m bps having translational entropy cost instead of loop
            )
    
 
    
# =============================================================================
# We're not getting good agreement for close to zero and positive values of F
# Here I make a few functions to try to understand why.
# First: If we exactly calculate g, do we then recover the curve? 
# =============================================================================
import decimal

def make_decimal_if_high_prec(x, high_prec):
  if high_prec:
    if type(x) == float:
      return(decimal.Decimal.from_float(x))
    else:
      return(decimal.Decimal(x))
  return(x)

def high_prec_log(high_prec):
  if high_prec:  # this is how you take the log of a decimal.Decimal class
    log = lambda x : x.ln()
  else:
    log = lambda x: np.log(float(x))
  return(log)

def high_prec_exp(high_prec):
  if high_prec:  # this is how you take the log of a decimal.Decimal class
    exp = lambda x : x.exp()
  else:
    exp = lambda x: np.exp(float(x))
  return(exp)


def get_integer_partitions(x, n, include_zero=True):
  # x is number to split up
  # n is the number of integers into which to split it up
  # Get all orderings of each partition

  # From https://stackoverflow.com/questions/58915599/generate-restricted-weak-integer-compositions-or-partitions-of-an-integer-n-in
  def constrained_partitions(n, k, min_elem, max_elem):
      allowed = range(max_elem, min_elem-1, -1)

      def helper(n, k, t):
          if k == 0:
              if n == 0:
                  yield t
          elif k == 1:
              if n in allowed:
                  yield t + (n,)
          elif min_elem * k <= n <= max_elem * k:
              for v in allowed:
                  yield from helper(n - v, k - 1, t + (v,))

      return helper(n, k, ())
  if include_zero:
    return(list(constrained_partitions(x, n, 0, x)))
  return(list(constrained_partitions(x, n, 1, x)))


def stirling_factorial(x):
  if x <= 1:  # x == 0 or x == 1:
    return(1)
  if x < 0:
    return(0)
  return(np.sqrt(2 * np.pi * x) * (x / np.exp(1))**x)

def factorial_fxn(stirling_approx, high_prec):
  if stirling_approx:
    fact = lambda x : stirling_factorial(x)
  else:
    def fact(x):
      if x < 0:
        return(1)
      return(gamma(x + 1))  # enables x to not be an integer
      #return(np.math.factorial(x))

  if high_prec:  # if we need very high precision
    fact_prec = lambda x: decimal.Decimal(fact(x))
  else:
    fact_prec = lambda x: fact(x)

  return(fact_prec)

def factorial_res(res, stirling_approx, high_prec):
  # Given the result of a ratio of factorials, get it to look acceptable
  if not stirling_approx:
    if high_prec:
      r = decimal.Decimal("0.")  # to get rid of floating point errors
      return(res.quantize(r))
    else:
      return(int(res))
  else:
    if high_prec:
      r = decimal.Decimal(".01")  # truncate at some point
      return(res.quantize(r))
    else:
      return(res)
  
def Nnp_with_neigh(n, Np, stirling_approx=False, high_prec=False): 
  # Gives the number of ways of making Np bonds with n stickers disallowing pseudoknots but allowing neighbor bonds

  fact = factorial_fxn(stirling_approx, high_prec)

  num = fact(n)
  den = fact(n-(2*Np)) * fact(Np+1) * fact(Np)
  res = num / den

  return(factorial_res(res, stirling_approx, high_prec))

def Nnp_no_neigh(n, Np, stirling_approx=False, high_prec=False):
  # Same as orig_eqn but disallowing neighbor bonds -- only applicable to monomers
  fact = factorial_fxn(stirling_approx, high_prec)

  num = fact(n - Np) * fact(n - Np - 1)
  den = fact(n - 2*Np) * fact(n - 2*Np - 1) * fact(Np+1) * fact(Np)
  res = num / den

  return(factorial_res(res, stirling_approx, high_prec))

def binomial(n, k, stirling_approx=False, high_prec=False):
  if n < k:
    return(0)
    
  fact = factorial_fxn(stirling_approx, high_prec)

  num = fact(n)
  den = fact(n - k) * fact(k)
  res = num / den

  return(factorial_res(res, stirling_approx, high_prec))

def t(n, x, stirling_approx=False, high_prec=False):
  if n < 2 * x or x <= 0 :
    return(0)
  return(binomial(n - x, x, stirling_approx=stirling_approx, high_prec=high_prec))

def sum_t(n, x, q, stirling_approx=False, high_prec=False): 
  # q is index for recursion
  if q == 1:
    return(t(n, x, stirling_approx=stirling_approx, high_prec=high_prec))
  res = np.sum([t(n, i, stirling_approx=stirling_approx, high_prec=high_prec) * 
                 sum_t(n, x-i, q-1, stirling_approx=stirling_approx, high_prec=high_prec) 
                 for i in range(1, x - q + 2)])
  return(res)

def t_prime(n, m, x, stirling_approx=False, high_prec=False):
  return(np.sum([binomial(m, q, stirling_approx=stirling_approx, high_prec=high_prec) *
                  sum_t(n, x, q, stirling_approx=stirling_approx, high_prec=high_prec)
                  for q in range(1, x+1)]))

def Nnp_no_neigh_multimer(n, m, Np, stirling_approx=False, high_prec=False):
  # Disallowing neighbor binding -- gives actual answer for m strands, not correcting for disconnected m-mers
  res = Nnp_with_neigh(n * m, Np, stirling_approx=stirling_approx, high_prec=high_prec)
  if Np == 0:
    return(res)  # no need to perform sum
  if m == 1:  # can speed up significantly
    return(Nnp_no_neigh(n, Np, stirling_approx=stirling_approx, high_prec=high_prec))

  sum_list = [(-1)**x * t_prime(n, m, x, stirling_approx=stirling_approx, high_prec=high_prec) * 
                  Nnp_with_neigh(n * m - 2*x, Np - x, stirling_approx=stirling_approx, high_prec=high_prec)
                  for x in range(1, Np + 1)]
  res += np.sum(sum_list)
  return(res)

max_n = 70
max_m = 15
max_Np = int(max_n * max_m / 2)

num_ways_to_make_connected_mmer_with_neigh = np.zeros((max_n, max_m, max_Np)) - 1  # initialize to -1
num_ways_to_make_connected_mmer_no_neigh = np.zeros((max_n, max_m, max_Np)) - 1  # initialize to -1
num_ways_to_make_connected_mmer_with_neigh_prec = np.zeros((max_n, max_m, max_Np), dtype=decimal.Decimal) - 1  # initialize to -1
num_ways_to_make_connected_mmer_no_neigh_prec = np.zeros((max_n, max_m, max_Np), dtype=decimal.Decimal) - 1  # initialize to -1

def num_ways_to_make_connected_mmer_fxn(n, m, Np, allow_neighbor_binding, high_prec=False):
  if allow_neighbor_binding:
    if high_prec:
      num_ways_to_make_connected_mmer = num_ways_to_make_connected_mmer_with_neigh_prec
    else:
      num_ways_to_make_connected_mmer = num_ways_to_make_connected_mmer_with_neigh
  else:
    if high_prec:
      num_ways_to_make_connected_mmer = num_ways_to_make_connected_mmer_no_neigh_prec
    else:
      num_ways_to_make_connected_mmer = num_ways_to_make_connected_mmer_no_neigh


  if num_ways_to_make_connected_mmer[n, m, Np] != -1:
    return(num_ways_to_make_connected_mmer[n, m, Np])

  if Np < m - 1 or Np > n * m / 2:
    zero = make_decimal_if_high_prec(0, high_prec)
    num_ways_to_make_connected_mmer[n, m, Np] = zero
    return(zero)


  if allow_neighbor_binding:
    uncorrected = Nnp_with_neigh(n * m, Np, stirling_approx=False, high_prec=high_prec)
  else:
    uncorrected = Nnp_no_neigh_multimer(n, m, Np, stirling_approx=False, high_prec=high_prec)

  if m == 1:
    num_ways_to_make_connected_mmer[n, m, Np] = uncorrected
    return(uncorrected)
  
  unconnected_possibilites, unconnected_counts = ways_to_connect_m(m)

  correction = make_decimal_if_high_prec(0, high_prec)
  for e, unconnected_possibility in enumerate(unconnected_possibilites):
    integer_partitions = get_integer_partitions(Np, len(unconnected_possibility), include_zero=True)
    # print(integer_partitions, unconnected_possibility)
    for Np_partition in integer_partitions:
      Np_partition_correction = make_decimal_if_high_prec(1, high_prec)
      unconnected_counts[e] = make_decimal_if_high_prec(unconnected_counts[e], high_prec)

      for e2, mp in enumerate(unconnected_possibility):
        # print(Np_partition, Np_partition[e2], mp)
        curr_correction = make_decimal_if_high_prec(
            num_ways_to_make_connected_mmer_fxn(
                n, mp, Np_partition[e2], 
                allow_neighbor_binding=allow_neighbor_binding, high_prec=high_prec), 
                high_prec)
        Np_partition_correction *= curr_correction
      correction += unconnected_counts[e] * Np_partition_correction
  
  corrected = uncorrected - correction
  num_ways_to_make_connected_mmer[n, m, Np] = corrected # / m
  
  return(corrected)



def analytical_simp_predicted_logZ_few_bonds_exact(
        n, m, eff_S_loop, adjacent_stickers_can_bind, FE_from_one_bp=-10, 
        DNA=False, T=273.15+37, max_num_bonds=2, high_prec=False):
    if DNA:
        E_cost = 1.96
    else:
        E_cost = 4.09
    translational_free_energy_cost_of_binding = E_cost - kB * T * np.log(rhoH20)

    loop_ent = closedNet0EntropyFxn(eff_S_loop, g)
    beta_eff_Fb = (FE_from_one_bp - T * loop_ent)/(kB * T)
    ent_trans_diff = -translational_free_energy_cost_of_binding/T - loop_ent

    last_few_g = [num_ways_to_make_connected_mmer_fxn(
        n, m + 1, Np, adjacent_stickers_can_bind, high_prec=high_prec)
        for Np in range(max_num_bonds + 1)]
    # print(last_few_g)
    log = high_prec_log(high_prec)
    exp = high_prec_exp(high_prec)
    last_few_terms = [exp(log(last_few_g[Np]) - make_decimal_if_high_prec(
        Np * beta_eff_Fb, high_prec)) for Np in range(max_num_bonds + 1)
        if last_few_g[Np] > 0]
    # print(last_few_terms)
    return(m * ent_trans_diff/kB - np.log(m + 1) + float(log(np.sum(last_few_terms))))


def NpStarFxn(n, m, eff_S_loop, adjacent_stickers_can_bind, FE_from_one_bp, 
           alpha_0=0.42, alpha_n=0):
    loop_ent = closedNet0EntropyFxn(eff_S_loop, g)
    beta_eff_Fb = (FE_from_one_bp - T * loop_ent)/(kB * T)

    alpha = alpha_0 + n * alpha_n  #n/50 + 0.25 + beta_eff_Fb / 50 # 0.42
    num_stickers = (m + 1) * n + (1-adjacent_stickers_can_bind) * m * alpha  # since m is # strands - 1

    if adjacent_stickers_can_bind:
        NpStar = num_stickers / (2 + np.exp(beta_eff_Fb / 2))
    else:
        NpStar = ((num_stickers-1) / 2) * (1 - np.exp(beta_eff_Fb / 4) / (np.sqrt(4 + np.exp(beta_eff_Fb / 2))))
        
    return(NpStar)






# Constrain concentrations to all be positive by solving in log-space
def cons_law_old(multimer_log_concs, conc_total):
    # multimer_log_concs is a length-m array giving the log-concentration of each m-mer
    # Returns N0 - (N_1 + 2 N_2 + 3 N_3 + ... + m N_m)
    return(conc_total - np.sum(np.arange(1, 1 + len(multimer_log_concs)) * 
                              np.exp(multimer_log_concs)))


def cons_law(multimer_log_concs, conc_total):
    # multimer_log_concs is a length-m array giving the log-concentration of each m-mer
    # Returns N0 - (N_1 + 2 N_2 + 3 N_3 + ... + m N_m)
    return(np.log(conc_total) - log_sum_exp(
        np.log(np.arange(1, 1 + len(multimer_log_concs))) + multimer_log_concs, 
        boltzmann_offset=-20))


def eq_Z_ratios(multimer_log_concs, m, n, corrected_log_Z):
    # Returns log(N_1^m / Z_1^m) - log(N_m rhoH20^(m-1) / Z_m) for the m specified
    # Working in log space is easier 
    # m is actually m+1 (i.e. m=1 is dimers, m=2 is trimers, etc.)
    
    if len(corrected_log_Z.shape) == 1:  # that means it's for a particular n already
        return((m+1) * (multimer_log_concs[0] - corrected_log_Z[0]) - (
            multimer_log_concs[m] - corrected_log_Z[m] 
            + (m+1-1) * np.log(rhoH20)))

    return((m+1) * (multimer_log_concs[0] - corrected_log_Z[0, n]) - (
            multimer_log_concs[m] - corrected_log_Z[m, n] 
            + (m+1-1) * np.log(rhoH20)))


def canonical_eqs(multimer_log_concs, n, conc_total, corrected_log_Z):
    return(np.array(
            [cons_law(multimer_log_concs, conc_total=conc_total),] + 
            [eq_Z_ratios(multimer_log_concs, m=m, n=n, corrected_log_Z=corrected_log_Z) 
            for m in range(1, len(multimer_log_concs))]
            ))

def get_eq_multimer_concs(corrected_log_Z, conc, max_num_guesses=100):
    max_m, max_n = np.shape(corrected_log_Z)
    
    # multimer_log_concs_guess = np.array([np.log(conc)] + [-10] * (max_m - 1))
    # multimer_log_concs_guess = np.array([-10] * (max_m - 1) + [np.log(conc)])
    # multimer_log_concs_guess = np.array([np.log(conc)] * max_m) / np.arange(1, max_m + 1)
    
    equilibrium_multimer_concentrations = np.zeros((max_m, max_n))
    
    for n in range(2, max_n):

        for guess in range(max_num_guesses):
            if guess == 0:
                multimer_log_concs_guess = np.array([np.log(conc)] + [-20] * (max_m - 1))
           
            elif guess == 1:
                multimer_log_concs_guess = [np.log(conc)] * max_m
                
            elif guess == 2:
                multimer_log_concs_guess = np.array([-30] * (max_m - 1) + [np.log(conc)])
        
            else:
                multimer_log_concs_guess = np.log(  # agnostic guess
                    np.random.random(max_m) * conc / (
                        max_m / 2 * (np.arange(max_m) + 1)))
        
        
            if n >= 2:
                equilibrium_multimer_concentrations[:, n] = np.exp(fsolve(
                    canonical_eqs, multimer_log_concs_guess, args=(n, conc, corrected_log_Z),
                    # maxfev=(100*(max_m+1) * 20 ), xtol=1.49012e-08 / 20#, factor=0.1
                    ))
            else:  # never gets called -- just thought it might be better, but turns out not to be.
                equilibrium_multimer_concentrations[:, n] = np.exp(fsolve(
                    canonical_eqs, np.log(equilibrium_multimer_concentrations[:, n - 1]), 
                    args=(n, conc, corrected_log_Z),
                    # maxfev=(100*(max_m+1) * 20 ), xtol=1.49012e-08 / 20#, factor=0.1
                    ))
            
            # Check that we were able to satisfy the equations
            equilibrium_multimer_concentrations_check = np.zeros((max_m, max_n))
            equilibrium_multimer_concentrations_check[:, n] = canonical_eqs(
                    np.log(equilibrium_multimer_concentrations[:, n]), n, 
                    conc_total=conc, corrected_log_Z=corrected_log_Z)
           
            if not np.all(np.isclose(equilibrium_multimer_concentrations_check, 0)):
                if guess == max_num_guesses - 1:
                    print('Was not able to satisfy all the equations simultaneously ')
            else:
                break

    return(equilibrium_multimer_concentrations)



# =============================================================================
# =============================================================================
# =============================================================================
# # # Load variables
# =============================================================================
# =============================================================================
# =============================================================================

multimers_stored_log_boltzmann_factor_per_FE_dict = dict()
multimers_stored_log_boltzmann_factor_per_FE_dict[1] = dict()
multimers_stored_log_boltzmann_factor_per_FE_dict[4] = dict()

ns_loaded_per_FE_dict = dict()
ns_loaded_per_FE_dict[1] = dict()
ns_loaded_per_FE_dict[4] = dict()

FE_from_one_bp_list_dict = dict()
FE_from_one_bp_list_dict[4] = [-15 + i for i in range(3)] + [-12 + i/2 for i in range(14)] + [-5 + i for i in range(7)]
FE_from_one_bp_list_dict[1] = [-15 + i for i in range(17)] 

corrected_log_Z_per_FE_dict = dict()
complexes_per_FE_dict = dict()

len_repeat_bp = 2  # number of nts comprising each complementary section of repeat
T = 273.15 + 37
max_m = 15

for len_linker in [1, 4]:  # number of nts comprising each non-complementary section

    for e, FE_from_one_bp in enumerate(FE_from_one_bp_list_dict[len_linker]):
        if FE_from_one_bp % 1 == 0:
            FE_from_one_bp_list_dict[len_linker][e] = int(FE_from_one_bp)
            
    
    for FE_from_one_bp in FE_from_one_bp_list_dict[len_linker]:
        folder_name_init = '/Users/your_folder_name_here/stored_partition_fxns/'
        folder_name = folder_name_init
        folder_name += 'len_repeat_bp_' + str(len_repeat_bp) + '/'
        folder_name += 'len_linker_' + str(len_linker) + '/'
        folder_name += 'T_' + str(T) + '/'
        folder_name += 'FE_from_one_bp_' + str(FE_from_one_bp) + '/'
        folder_name += 'b_' + str(b) + '/'
        folder_name += 'vs_' + str(vs) + '/'
            
        
        def loadFxnOnlyZ(max_m):
            start = time.time()
            try:
                multimers_stored_log_boltzmann_factor = load(folder_name + 'multimers_stored_log_boltzmann_factor')
                ns_loaded = load(folder_name + 'ns_loaded')
                if len(ns_loaded) < max_m:
                    ns_loaded += [0] * (max_m - len(ns_loaded))
            except:
                multimers_stored_log_boltzmann_factor = dict()
                ns_loaded = [0] * max_m
                
            print('Time to load variables = ' + str(time.time() - start))
            return(multimers_stored_log_boltzmann_factor, ns_loaded)
    
        
        (multimers_stored_log_boltzmann_factor_per_FE_dict[len_linker][FE_from_one_bp], 
         ns_loaded_per_FE_dict[len_linker][FE_from_one_bp]) = loadFxnOnlyZ(max_m)


    corrected_log_Z_per_FE_dict[len_linker] = np.zeros((len(FE_from_one_bp_list_dict[len_linker]), 
                                       max_m, max(ns_loaded_per_FE_dict[len_linker][FE_from_one_bp])))
    complexes_per_FE_dict[len_linker] = np.zeros((len(FE_from_one_bp_list_dict[len_linker]), 
                                 max_m, max(ns_loaded_per_FE_dict[len_linker][FE_from_one_bp])))
    for e, FE_from_one_bp in enumerate(FE_from_one_bp_list_dict[len_linker]):
        complexes_per_FE_dict[len_linker][e, :, :], corrected_log_Z_per_FE_dict[len_linker][e, :, :] = create_complexes_from_multimers(
            multimers_stored_log_boltzmann_factor_per_FE_dict[len_linker][FE_from_one_bp], 
            ns_loaded_per_FE_dict[len_linker][FE_from_one_bp], DNA=False)

    
    
    
    
# =============================================================================
# =============================================================================
# =============================================================================
# # # Make figures
# =============================================================================
# =============================================================================
# =============================================================================

# =============================================================================
# Z_m/Z_1^m as a function of binding energy
# =============================================================================
import matplotlib
matplotlib.rc('font', **{'size' : 22})

m_to_plot = 3
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True) # plt.figure()
axs = [ax1, ax2]
for e, len_linker in enumerate([4, 1]):
    #plt.subplot(1 + e, 2, 1)
    x_axis = FE_from_one_bp_list_dict[len_linker]
    for n in [3, 4, 5, 6, 7, 8]: # range(2, ns_loaded_per_FE_4[-10][m_to_plot]): # [4, 5, 6, 7, 8, 9]: #
        if n%2 == 0:
            linestyle='-'
        else:
            linestyle='--'
        axs[e].plot(x_axis, 
                 np.exp(corrected_log_Z_per_FE_dict[len_linker][:, m_to_plot , n] - 
                        (m_to_plot + 1) * corrected_log_Z_per_FE_dict[len_linker][:, 0, n]), 
                 label='n = ' + str(n), linestyle=linestyle)
    if e == 1:
        axs[e].legend(bbox_to_anchor=(1.05, 1))
    axs[e].set_xlabel(r'$F_b$' + ' (kcal/mol)')
    if e == 0:
        axs[e].set_ylabel(r'$Z_4 / Z_1^4$')
    #plt.ylabel(r'$\log(Z_4) - 4 \log(Z_1)$')
    axs[e].set_yscale('log')
    axs[e].set_yticks([1e0, 1e10, 1e20, 1e30])
    # else:
    #     axs[e].set_yticks([1e0, 1e10, 1e20, 1e30], ['', '', '', ''])
    axs[e].set_ylim([1e-5, 1e35])
    axs[e].set_xticks([-15, -10, -5, 0])
    if len_linker >= 3:
        axs[e].set_title('Allowing \n neighbor bonds', fontsize=18)
    else:
        axs[e].set_title('Disallowing \n neighbor bonds', fontsize=18)
plt.show()






matplotlib.rc('font', **{'size' : 20})

m_to_plot = 4
# fig, axs_full = plt.subplots(2, 2, sharex=True, sharey=True) # plt.figure()
# axs = axs_full.flat
fig = plt.figure()
gs = fig.add_gridspec(2, 2, wspace=0.15, hspace=0.25)
axs_full = gs.subplots(sharex=True, sharey=True)
axs = axs_full.flat
for e2, n_to_plot in enumerate([18, 19]):
    for e1, len_linker in enumerate([4, 1]):
        e = 2 * e2 + e1
        Np_pred_monomer = [NpStarFxn(n_to_plot, 0, 4, len_linker >= 3, FE_from_one_bp, alpha_0=0.42, alpha_n=0)
            for FE_from_one_bp in FE_from_one_bp_list_dict[len_linker]]
        Np_pred_multimer = [NpStarFxn(n_to_plot, m_to_plot, 4, len_linker >= 3, FE_from_one_bp, alpha_0=0.42, alpha_n=0)
            for FE_from_one_bp in FE_from_one_bp_list_dict[len_linker]]
        
        loop_len_pred_monomer = [min(
            len_linker * (n_to_plot-2), 
            (4.3 + 2.7*(len_linker==4)) * n_to_plot / (2 * (Np))) for Np in Np_pred_monomer]
        loop_len_pred_multimer = [min(
            len_linker * (n_to_plot-2), 
            (4.3 + 2.7*(len_linker==4)) * n_to_plot *(m_to_plot + 1) / (2 * (Np))) for Np in Np_pred_multimer]
        # loop_len_pred_monomer = [eff_S_loop for Np in Np_pred_monomer]
        # loop_len_pred_multimer = [eff_S_loop for Np in Np_pred_monomer]
        

        color = '#1E88E5' #next(axs[e]._get_lines.prop_cycler)['color']
        axs[e].plot(FE_from_one_bp_list_dict[len_linker], 
                 np.exp(corrected_log_Z_per_FE_dict[len_linker][:, m_to_plot , n_to_plot] - 
                 (m_to_plot + 1) * corrected_log_Z_per_FE_dict[len_linker][:, 0, n_to_plot]), 
                 label='True', color=color, linewidth=3, alpha=0.6)
        
        # color = next(ax._get_lines.prop_cycler)['color']
        # plt.plot(FE_from_one_bp_list, 
        #          [analytical_simp_predicted_logZ_few_bonds_exact(
        #              n_to_plot, m_to_plot, loop_len_pred_multimer[j], len_linker >= 3, 
        #              i, DNA=False, max_num_bonds=int(n * (m_to_plot+1) / 2)) - 
        #          (m_to_plot + 1) * analytical_simp_predicted_logZ_few_bonds_exact(
        #              n_to_plot, 0, loop_len_pred_monomer[j], len_linker >= 3, 
        #              i, DNA=False, max_num_bonds=int(n/2)) 
        #          for j, i in enumerate(FE_from_one_bp_list)],
        #          label='n = ' + str(n) + '; full sum', color=color, linestyle='--')
        
        # color = next(axs[e]._get_lines.prop_cycler)['color']
        # axs[e].plot(FE_from_one_bp_list_dict[len_linker], 
        #          [np.exp(analytical_simp_predicted_logZ_all_bonds(
        #              n_to_plot, m_to_plot, loop_len_pred_multimer[j], len_linker >= 3, 
        #              i, DNA=False, alpha_0=0.42, alpha_n=0) - 
        #          (m_to_plot + 1) * analytical_simp_predicted_logZ_all_bonds(
        #              n_to_plot, 0, loop_len_pred_multimer[j], len_linker >= 3, 
        #              i, DNA=False) )
        #          for j, i in enumerate(FE_from_one_bp_list_dict[len_linker])], 
        #          label='Pred: low E', color=color, linestyle=':')
        
        color = '#004D40' #next(axs[e]._get_lines.prop_cycler)['color']
        axs[e].plot([i for i in FE_from_one_bp_list_dict[len_linker] if i < -2], 
                  [np.exp(analytical_simp_predicted_logZ_some_bonds(
                      n_to_plot, m_to_plot, loop_len_pred_multimer[j], len_linker >= 3, 
                      i, DNA=False, alpha_0=0.42, alpha_n=0, include_saddlepoint=True) - 
                  (m_to_plot + 1) * analytical_simp_predicted_logZ_all_bonds(
                      n_to_plot, 0, loop_len_pred_multimer[j], len_linker >= 3, 
                      i, DNA=False) )
                  for j, i in enumerate(FE_from_one_bp_list_dict[len_linker]) if i < -2], 
                  label='Strong', color=color, linestyle=':', linewidth=2.5)
        
        color = '#FFC107' #next(axs[e]._get_lines.prop_cycler)['color']
        axs[e].plot(FE_from_one_bp_list_dict[len_linker], 
                 [np.exp(analytical_simp_predicted_logZ_some_bonds(
                     n_to_plot, m_to_plot, loop_len_pred_multimer[j], len_linker >= 3, 
                     i, DNA=False, alpha_0=0.42, alpha_n=0, include_saddlepoint=True) - 
                 (m_to_plot + 1) * analytical_simp_predicted_logZ_some_bonds(
                     n_to_plot, 0, loop_len_pred_multimer[j], len_linker >= 3, 
                     i, DNA=False, include_saddlepoint=True) )
                 for j, i in enumerate(FE_from_one_bp_list_dict[len_linker])], 
                 label='Intermediate', 
                 color=color, linestyle=':', linewidth=2.5)  #linestyle='--', linewidth=1.5)
    
        color = '#D81B60' #next(axs[e]._get_lines.prop_cycler)['color']
        axs[e].plot([i for i in FE_from_one_bp_list_dict[len_linker] if i > -8], 
                 [np.exp(analytical_simp_predicted_logZ_few_bonds_exact(
                     n_to_plot, m_to_plot, loop_len_pred_multimer[j], len_linker >= 3, 
                     i, DNA=False, max_num_bonds=m_to_plot+2) - 
                 (m_to_plot + 1) * analytical_simp_predicted_logZ_few_bonds_exact(
                     n_to_plot, 0, loop_len_pred_multimer[j], len_linker >= 3, 
                     i, DNA=False, max_num_bonds=2) )
                 for j, i in enumerate(FE_from_one_bp_list_dict[len_linker]) if i > -8], 
                 label='Weak', 
                 color=color, linestyle=':', linewidth=2.5)  #linestyle='-.', linewidth=1.5)
        
        # Plot predicted crossover points
        # color = next(axs[e]._get_lines.prop_cycler)['color']
        axs[e].plot([2 * np.log(4 / (n_to_plot-2)) +   # when NbStar = n/2 - 1 for monomers
                  T * closedNet0EntropyFxn(loop_len_pred_multimer[0], g)] * 2,
                  [0, 1e40],
                  '--', color='k', linewidth=0.5)
        axs[e].plot([2 * np.log((n_to_plot -6) / 3) +  # when NbStar = 3 for monomers
                  T * closedNet0EntropyFxn(loop_len_pred_multimer[-5], g)] * 2,
                  [0, 1e40],
                  '--', color='k', linewidth=0.5)
        
        
        # if len_linker >= 3:
        #     axs[e].set_title('n = ' + str(n_to_plot) + '\n with neighbor bonds' + '   '*(e==0), fontsize=15)
        # else:
        #     axs[e].set_title('n = ' + str(n_to_plot) + '\n no neighbor bonds', fontsize=15)
        if e < 2:
            if len_linker >= 3:
                axs[e].set_title('Allowing \n neighbor bonds', fontsize=18)
            else:
                axs[e].set_title('Disallowing \n neighbor bonds', fontsize=18)
        axs[e].text(-15, 1e4, 'n = ' + str(n_to_plot), fontsize=12, bbox=dict(alpha=0.5, facecolor='white'))
        
        
        axs[e].set_xlabel(r'$F_b$' + ' (kcal/mol)')
        axs[e].set_ylabel(r'$Z_{} / Z_1^{}$'.format(m_to_plot + 1, m_to_plot + 1))
        #plt.ylabel(r'$\log(Z_4) - 4 \log(Z_1)$')
        axs[e].set_yscale('log')
        axs[e].set_yticks([1e0, 1e20, 1e40])
        axs[e].set_ylim([1e-1, 1e41])
        axs[e].set_xticks([-15, -10, -5, 0])

for ax in axs:
    ax.label_outer()    
axs[1].legend(bbox_to_anchor=(1.05, 1), fontsize=12)
plt.show()


# =============================================================================
# Plot Z as a function of n
# =============================================================================

matplotlib.rc('font', **{'size' : 22})

ns_to_plot = range(2, 21)
for FE_to_plot in [-10, -6]:
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True) # plt.figure()
    axs = [ax1, ax2]
    for e, len_linker in enumerate([4, 1]):
        FE_index = np.where(np.array(FE_from_one_bp_list_dict[len_linker])==FE_to_plot)[0][0]
        
        for m_to_plot in [0, 1, 2, 3, 5, 7, 9]: # range(2, ns_loaded_per_FE_4[-10][m_to_plot]): # [4, 5, 6, 7, 8, 9]: #

            Np_pred_monomer = [NpStarFxn(n, 0, 4, len_linker >= 3, FE_to_plot, alpha_0=0.42, alpha_n=0)
                for n in ns_to_plot]
            Np_pred_multimer = [NpStarFxn(n, m_to_plot, 4, len_linker >= 3, FE_to_plot, alpha_0=0.42, alpha_n=0)
                for n in ns_to_plot]
            
            loop_len_pred_monomer = [
                (2 + 3*(len_linker==4)) * n / (2 * (Np_pred_monomer[j])) for j, n in enumerate(ns_to_plot)]
            loop_len_pred_multimer = [
                (2 + 3*(len_linker==4)) * n * (m_to_plot + 1) / (2 * (Np_pred_multimer[j])) for j, n in enumerate(ns_to_plot)]

            color = next(axs[e]._get_lines.prop_cycler)['color']
            axs[e].plot(ns_to_plot, 
                     np.exp(corrected_log_Z_per_FE_dict[len_linker][FE_index, m_to_plot , ns_to_plot[0]:ns_to_plot[-1] + 1] - 
                            (m_to_plot + 1) * corrected_log_Z_per_FE_dict[len_linker][FE_index, 0, ns_to_plot[0]:ns_to_plot[-1] + 1]), 
                     label='m = ' + str(m_to_plot), linestyle='-', color=color)
            
            if FE_to_plot < -8:
                if m_to_plot < 2:
                    axs[e].plot(ns_to_plot, 
                              np.exp([analytical_simp_predicted_logZ_all_bonds(
                                  n, m_to_plot, loop_len_pred_multimer[j], len_linker >= 3, 
                                  FE_to_plot, DNA=False, alpha_0=0.42, alpha_n=0) - 
                                    (m_to_plot + 1) * analytical_simp_predicted_logZ_all_bonds(
                                  n, 0, loop_len_pred_multimer[j], len_linker >= 3, 
                                  FE_to_plot, DNA=False, alpha_0=0.42, alpha_n=0) 
                                    for j, n in enumerate(ns_to_plot)]), 
                              linestyle=':', color=color)
                else:
                    # print(loop_len_pred_multimer, loop_len_pred_monomer)
                    axs[e].plot(ns_to_plot, 
                             np.exp([analytical_simp_predicted_logZ_some_bonds(
                                 n, m_to_plot, loop_len_pred_multimer[j] , len_linker >= 3, 
                                 FE_to_plot, DNA=False, alpha_0=0.42, alpha_n=0, include_saddlepoint=True) - 
                                    (m_to_plot + 1) * analytical_simp_predicted_logZ_all_bonds(
                                 n, 0, loop_len_pred_multimer[j] , len_linker >= 3, 
                                 FE_to_plot, DNA=False, alpha_0=0.42, alpha_n=0) 
                                    for j, n in enumerate(ns_to_plot)]), 
                             linestyle=':', color=color)
            else:
                axs[e].plot(ns_to_plot, 
                         np.exp([analytical_simp_predicted_logZ_some_bonds(
                             n, m_to_plot, loop_len_pred_multimer[j], len_linker >= 3, 
                             FE_to_plot, DNA=False, alpha_0=0.42, alpha_n=0, include_saddlepoint=True) - 
                                (m_to_plot + 1) * analytical_simp_predicted_logZ_some_bonds(
                             n, 0, loop_len_pred_multimer[j], len_linker >= 3, 
                             FE_to_plot, DNA=False, alpha_0=0.42, alpha_n=0, include_saddlepoint=True) 
                                for j, n in enumerate(ns_to_plot)]), 
                         linestyle=':', color=color)
    
        if e == 1:
            axs[e].legend(bbox_to_anchor=(1.05, 1))
        axs[e].set_xlabel(r'$n$')
        axs[e].set_ylabel(r'$Z_m / Z_1^m$')
        axs[e].set_yscale('log')
        axs[e].set_yticks([1e0, 1e20, 1e40, 1e60])
        axs[e].set_ylim([1e-1, 1e61])
        axs[e].set_xticks([5, 10, 15, 20])
        if len_linker >= 3:
            axs[e].set_title('Allowing \n neighbor bonds', fontsize=18)
        else:
            axs[e].set_title('Disallowing \n neighbor bonds', fontsize=18)
    for ax in axs:
        ax.label_outer()    
    plt.show()





ns_to_plot = range(2, 21)
for FE_to_plot in [-10, -6]:
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True) # plt.figure()
    axs = [ax1, ax2]
    for e, len_linker in enumerate([4, 1]):
        FE_index = np.where(np.array(FE_from_one_bp_list_dict[len_linker])==FE_to_plot)[0][0]
        
        for m_to_plot in [0, 1, 2, 3, 5, 7, 9]: # range(2, ns_loaded_per_FE_4[-10][m_to_plot]): # [4, 5, 6, 7, 8, 9]: #

            Np_pred_monomer = [NpStarFxn(n, 0, 4, len_linker >= 3, FE_to_plot, alpha_0=0.42, alpha_n=0)
                for n in ns_to_plot]
            Np_pred_multimer = [NpStarFxn(n, m_to_plot, 4, len_linker >= 3, FE_to_plot, alpha_0=0.42, alpha_n=0)
                for n in ns_to_plot]
            
            loop_len_pred_monomer = [
                (4.3 + 2.7*(len_linker==4)) * n / (2 * (Np_pred_monomer[j])) for j, n in enumerate(ns_to_plot)]
            loop_len_pred_multimer = [
                (4.3 + 2.7*(len_linker==4)) * n * (m_to_plot + 1) / (2 * (Np_pred_multimer[j])) for j, n in enumerate(ns_to_plot)]

            color = next(axs[e]._get_lines.prop_cycler)['color']
            axs[e].plot(ns_to_plot, 
                     np.exp(corrected_log_Z_per_FE_dict[len_linker][FE_index, m_to_plot , ns_to_plot[0]:ns_to_plot[-1] + 1]), 
                     label='m = ' + str(m_to_plot), linestyle='-', color=color)
            
            if FE_to_plot < -8:
                if m_to_plot < 2:
                    axs[e].plot(ns_to_plot, 
                              np.exp([analytical_simp_predicted_logZ_all_bonds(
                                  n, m_to_plot, loop_len_pred_multimer[j], len_linker >= 3, 
                                  FE_to_plot, DNA=False, alpha_0=0.42, alpha_n=0)
                                    for j, n in enumerate(ns_to_plot)]), 
                              linestyle=':', color=color)
                else:
                    # print(loop_len_pred_multimer, loop_len_pred_monomer)
                    axs[e].plot(ns_to_plot, 
                             np.exp([analytical_simp_predicted_logZ_some_bonds(
                                 n, m_to_plot, loop_len_pred_multimer[j], len_linker >= 3, 
                                 FE_to_plot, DNA=False, alpha_0=0.42, alpha_n=0, include_saddlepoint=True)
                                    for j, n in enumerate(ns_to_plot)]), 
                             linestyle=':', color=color)
            else:
                axs[e].plot(ns_to_plot, 
                         np.exp([analytical_simp_predicted_logZ_some_bonds(
                             n, m_to_plot, loop_len_pred_multimer[j], len_linker >= 3, 
                             FE_to_plot, DNA=False, alpha_0=0.42, alpha_n=0, include_saddlepoint=True)
                                for j, n in enumerate(ns_to_plot)]), 
                         linestyle=':', color=color)
    
        if e == 1:
            axs[e].legend(bbox_to_anchor=(1.05, 1))
        axs[e].set_xlabel(r'$n$')
        axs[e].set_ylabel(r'$Z_m$')
        axs[e].set_yscale('log')
        axs[e].set_yticks([1e0, 1e100, 1e200, 1e300])
        axs[e].set_ylim([1e-1, 1e301])
        axs[e].set_xticks([5, 10, 15, 20])
        if len_linker >= 3:
            axs[e].set_title('Allowing \n neighbor bonds', fontsize=18)
        else:
            axs[e].set_title('Disallowing \n neighbor bonds', fontsize=18)
    for ax in axs:
        ax.label_outer()    
    plt.show()



# =============================================================================
# Combining the above plots
# =============================================================================
matplotlib.rc('font', **{'size' : 18})
for FE_to_plot in [-10, -6]:
    ns_to_plot = range(2, 21)

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, wspace=0.15, hspace=0.25)
    axs_full = gs.subplots(sharex='col', sharey='row')
    axs = axs_full.flat
    for e1, len_linker in enumerate([4, 1]):
        for e2, ratio in enumerate([0, 1]):
            e = 2 * e2 + e1
            FE_index = np.where(np.array(FE_from_one_bp_list_dict[len_linker])==FE_to_plot)[0][0]
            
            for m_to_plot in [0, 1, 2, 4, 6, 9]: # range(2, ns_loaded_per_FE_4[-10][m_to_plot]): # [4, 5, 6, 7, 8, 9]: #
    
                Np_pred_monomer = [NpStarFxn(n, 0, 4, len_linker >= 3, FE_to_plot, alpha_0=0.42, alpha_n=0)
                    for n in ns_to_plot]
                Np_pred_multimer = [NpStarFxn(n, m_to_plot, 4, len_linker >= 3, FE_to_plot, alpha_0=0.42, alpha_n=0)
                    for n in ns_to_plot]
                
                loop_len_pred_monomer = [
                    (4.3 + 2.7*(len_linker==4)) * n / (2 * (Np_pred_monomer[j])) for j, n in enumerate(ns_to_plot)]
                loop_len_pred_multimer = [
                    (4.3 + 2.7*(len_linker==4)) * n * (m_to_plot + 1) / (2 * (Np_pred_multimer[j])) for j, n in enumerate(ns_to_plot)]
    
                color = next(axs[e]._get_lines.prop_cycler)['color']
                axs[e].plot(ns_to_plot, 
                         np.exp(corrected_log_Z_per_FE_dict[len_linker][FE_index, m_to_plot , ns_to_plot[0]:ns_to_plot[-1] + 1] - 
                                ratio * (m_to_plot + 1) * 
                                corrected_log_Z_per_FE_dict[len_linker][FE_index, 0, ns_to_plot[0]:ns_to_plot[-1] + 1]), 
                         label=e * (str(m_to_plot + 1)), linestyle='-', color=color)
                
                if FE_to_plot < -8:
                    if m_to_plot < 2:
                        axs[e].plot(ns_to_plot, 
                                  np.exp([analytical_simp_predicted_logZ_all_bonds(
                                      n, m_to_plot, loop_len_pred_multimer[j], len_linker >= 3, 
                                      FE_to_plot, DNA=False, alpha_0=0.42, alpha_n=0) - 
                                        ratio * (m_to_plot + 1) * analytical_simp_predicted_logZ_all_bonds(
                                      n, 0, loop_len_pred_multimer[j], len_linker >= 3, 
                                      FE_to_plot, DNA=False, alpha_0=0.42, alpha_n=0) 
                                        for j, n in enumerate(ns_to_plot)]), 
                                  linestyle=':', color=color)
                    else:
                        # print(loop_len_pred_multimer, loop_len_pred_monomer)
                        axs[e].plot(ns_to_plot, 
                                 np.exp([analytical_simp_predicted_logZ_some_bonds(
                                     n, m_to_plot, loop_len_pred_multimer[j] , len_linker >= 3, 
                                     FE_to_plot, DNA=False, alpha_0=0.42, alpha_n=0, include_saddlepoint=True) - 
                                        ratio * (m_to_plot + 1) * analytical_simp_predicted_logZ_all_bonds(
                                     n, 0, loop_len_pred_multimer[j] , len_linker >= 3, 
                                     FE_to_plot, DNA=False, alpha_0=0.42, alpha_n=0) 
                                        for j, n in enumerate(ns_to_plot)]), 
                                 linestyle=':', color=color)
                else:
                    axs[e].plot(ns_to_plot, 
                             np.exp([analytical_simp_predicted_logZ_some_bonds(
                                 n, m_to_plot, loop_len_pred_multimer[j], len_linker >= 3, 
                                 FE_to_plot, DNA=False, alpha_0=0.42, alpha_n=0, include_saddlepoint=True) - 
                                    ratio * (m_to_plot + 1) * analytical_simp_predicted_logZ_some_bonds(
                                 n, 0, loop_len_pred_multimer[j], len_linker >= 3, 
                                 FE_to_plot, DNA=False, alpha_0=0.42, alpha_n=0, include_saddlepoint=True) 
                                    for j, n in enumerate(ns_to_plot)]), 
                             linestyle=':', color=color)
                
            if e == 0:
                axs[e].plot([-1],[-1],'k',label='True')
                axs[e].plot([-1],[-1],'k',linestyle=':', label='Predicted')
                axs[e].legend(loc='upper left', fontsize=12)
            if e == 1:
                axs[e].legend(bbox_to_anchor=(1.05, 1), ncol=1, fontsize=12, title=r'$m$')
            axs[e].set_xlabel(r'$n$')
            axs[e].set_xlim([1, ns_to_plot[-1] + 1])
            axs[e].set_yscale('log')
            if ratio == 1:
                axs[e].set_ylabel(r'$Z_m / Z_1^m$')
                axs[e].set_ylim([1e-1, 1e61])
                axs[e].set_yticks([1e0, 1e20, 1e40, 1e60])
            else:
                axs[e].set_ylabel(r'$Z_m$')
                axs[e].set_ylim([1e-1, 1e301])
                axs[e].set_yticks([1e0, 1e100, 1e200, 1e300])#, 1e300])
            axs[e].set_xticks([5, 10, 15, 20])
            if e < 2:
                if len_linker >= 3:
                    axs[e].set_title('Allowing \n neighbor bonds', fontsize=18)
                else:
                    axs[e].set_title('Disallowing \n neighbor bonds', fontsize=18)
    for ax in axs:
        ax.label_outer()
    # plt.tight_layout()
    plt.show()



# =============================================================================
# Concentrations
# =============================================================================
matplotlib.rc('font', **{'size' : 18})
def plot_monomers_oligomers_droplets_frac(
        frac_monomer_per_conc, frac_oligomer_per_conc, frac_droplet_per_conc, 
        xlabel=r'$c^{tot}$ ' + r'($\mu$M)', ylabel=r'$n$',
        yticks=[0, 16, 27], ytick_labels=[48 - 1, 48-17, 48 - 28],
        xticks=[0, (11 - 1) // 2, 11 - 1], xtick_labels=[r'$10$', r'$10^2$', r'$10^3$'],
        title='Fraction of RNA in a given phase', 
        plot_thirumalai=False, concs_to_try=[]):

    norm_0_1 = matplotlib.colors.Normalize(0,1)
    
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(8, 3), ncols=3)
    #plt.subplot(1,3,1)
    im1 = ax1.imshow(np.flipud(np.transpose(frac_monomer_per_conc)), 
                     norm=norm_0_1, aspect='auto')
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xtick_labels)
    ax1.set_yticks(yticks)  #[0, 10, 20])
    ax1.set_yticklabels(ytick_labels)  # [max_n_extrapolated, max_n_extrapolated - 10, max_n_extrapolated - 20])
    ax1.set_title('Monomers')
    ax1.set_ylabel(ylabel)
    
    im2 = ax2.imshow(np.flipud(np.transpose(frac_oligomer_per_conc)), 
                     norm=norm_0_1, aspect='auto')
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xtick_labels)
    ax2.set_yticks(yticks)  #[0, 10, 20])
    ax2.set_yticklabels([])  # [max_n_extrapolated, max_n_extrapolated - 10, max_n_extrapolated - 20])
    ax2.set_title('Oligomers')
    ax2.set_xlabel(xlabel)
    
    im3 = ax3.imshow(np.flipud(np.transpose(frac_droplet_per_conc)), 
                     norm=norm_0_1, aspect='auto')
    ax3.set_xticks(xticks)
    ax3.set_xticklabels(xtick_labels)
    ax3.set_yticks(yticks)  #[0, 10, 20])
    ax3.set_yticklabels([])  # [max_n_extrapolated, max_n_extrapolated - 10, max_n_extrapolated - 20])
    ax3.set_title('Aggregates')
    
    if plot_thirumalai:
        # def closest(lst, K):     
        #     idx = (np.abs(lst - K)).argmin()
        #     return idx

        # Get coordinates of each of their points in our plot
        thirumalai_concs = [20, 50, 100, 200, 400, 500]
        concs_uM = np.array(concs_to_try) * 1e6
        our_thirumalai_concs = [np.log(i/min(concs_uM)) * (len(concs_uM) - 1) / np.log(max(concs_uM)/min(concs_uM)) for i in thirumalai_concs]
        # our_thirumalai_concs_check = [closest(concs_uM, k) for k in thirumalai_concs]
        # print(our_thirumalai_concs, our_thirumalai_concs_check)
        thirumalai_ns = [20, 31, 47]
        # frac_droplet_per_conc has the first 2 n's missing compared to its definition outside the function
        our_thirumalai_ns = [frac_droplet_per_conc.shape[1] - i + 1 for i in thirumalai_ns]
        # print(frac_droplet_per_conc.shape, our_thirumalai_ns)  
        thirumalai_yes_x = [1, 1, 2, 2, 3, 3, 4]
        thirumalai_yes_y = [1, 2, 1, 2, 1, 2, 1]
        thirumalai_no_x = [0, 1, 2, 3, 5]
        thirumalai_no_y = [2, 0, 0, 0, 0]
    
        ax3.plot([our_thirumalai_concs[i] for i in thirumalai_yes_x],
                 [our_thirumalai_ns[i] for i in thirumalai_yes_y], 
                 '.', marker="o", markersize=5, markerfacecolor='b')
        ax3.plot([our_thirumalai_concs[i] for i in thirumalai_no_x],
                 [our_thirumalai_ns[i] for i in thirumalai_no_y], 
                 '.', marker="o", markersize=5, markerfacecolor='r')
    cb = fig.colorbar(im3, ax=ax3, label='Fraction')
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels([r'$0$', #r'$\frac{1}{2}$', 
                       r'$0.5$', r'$1$'])
    
    plt.suptitle(title)
    #fig.tight_layout()
    fig.subplots_adjust(top=0.7, bottom=0)
    plt.show()


FE_to_plot = -10
len_linker = 1
FE_index = np.where(np.array(FE_from_one_bp_list_dict[len_linker])==FE_to_plot)[0][0]
num_concs = 11  # 101  # make it an odd number
max_c = -3
max_m_for_oligomer = 4  # Thirumalai defines oligomers as 2-4 strands, droplets as anything larger


# First try just using the computational data
max_n_to_use = min([i for i in ns_loaded_per_FE_dict[len_linker][FE_to_plot]])
max_m_to_use = 10

concs_to_try = np.logspace(-5, max_c, num_concs)  # in units of M
frac_monomer_per_conc = np.zeros((num_concs, max_n_to_use))
frac_oligomer_per_conc = np.zeros((num_concs, max_n_to_use))
frac_droplet_per_conc = np.zeros((num_concs, max_n_to_use))

for e, conc in enumerate(concs_to_try):
    equilibrium_multimer_concentrations = get_eq_multimer_concs(
            corrected_log_Z_per_FE_dict[len_linker][FE_index, :max_m_to_use, :max_n_to_use], conc)
    
    # Compute what fraction of the RNA is in each phase
    frac_monomer_per_conc[e, :] = equilibrium_multimer_concentrations[0, :] / conc
    frac_oligomer_per_conc[e, :] = np.sum(np.transpose(equilibrium_multimer_concentrations[1:max_m_for_oligomer,:]) * 
                        np.arange(2, max_m_for_oligomer + 1), 1) / conc
    frac_droplet_per_conc[e, :] = np.sum(np.transpose(equilibrium_multimer_concentrations[max_m_for_oligomer:,:]) * 
                        np.arange(max_m_for_oligomer + 1, max_m_to_use + 1), 1) / conc


plot_monomers_oligomers_droplets_frac(
    frac_monomer_per_conc[:, 2:], frac_oligomer_per_conc[:, 2:], frac_droplet_per_conc[:, 2:], 
    title = 'Computational model', 
    yticks=[0, 5, 10, 15], 
    ytick_labels=[max_n_to_use - 1, max_n_to_use-6, max_n_to_use-11, max_n_to_use-16], 
    xticks=[0, (num_concs - 1) // 2, num_concs - 1],
    xtick_labels=[r'$10$', r'$10^2$', r'$10^3$'], )





# Now try using the extrapolated data
concs_to_try = np.logspace(-5, max_c, num_concs)  # in units of M
frac_monomer_per_conc = np.zeros((num_concs, max_n_to_use))
frac_oligomer_per_conc = np.zeros((num_concs, max_n_to_use))
frac_droplet_per_conc = np.zeros((num_concs, max_n_to_use))

log_Z_per_FE_pred = np.zeros((max_m_to_use, max_n_to_use))
for m in range(max_m_to_use):
    for n in range(2, max_n_to_use):
        if m < 2:
            log_Z_per_FE_pred[m, n] = analytical_simp_predicted_logZ_all_bonds(
                                  n, m, 4.3 + 2.7 * (len_linker==4), len_linker >= 3, 
                                  FE_to_plot, DNA=False, alpha_0=0.42, alpha_n=0)        
        else:
            log_Z_per_FE_pred[m, n] = analytical_simp_predicted_logZ_some_bonds(
                                  n, m, 4.3 + 2.7 * (len_linker==4), len_linker >= 3, 
                                  FE_to_plot, DNA=False, alpha_0=0.42, alpha_n=0, include_saddlepoint=True)


# Check the extrapolation
fig = plt.figure()
gs = fig.add_gridspec(2, 1, wspace=0.15, hspace=0.25)
axs_full = gs.subplots(sharey='row')
axs = axs_full.flat
ns_to_plot = range(2, max_n_to_use)
for e, ratio in enumerate([0, 1]):
    for m_to_plot in [0, 1, 2, 3, 5, 7, 9]: # range(2, ns_loaded_per_FE_4[-10][m_to_plot]): # [4, 5, 6, 7, 8, 9]: #
        color = next(axs[e]._get_lines.prop_cycler)['color']
        axs[e].plot(ns_to_plot, 
                 np.exp(corrected_log_Z_per_FE_dict[len_linker][FE_index, m_to_plot , ns_to_plot[0]:ns_to_plot[-1] + 1] - 
                        ratio * (m_to_plot + 1) * corrected_log_Z_per_FE_dict[len_linker][FE_index, 0, ns_to_plot[0]:ns_to_plot[-1] + 1]), 
                 label='m = ' + str(m_to_plot), linestyle='-', color=color)
        axs[e].plot(ns_to_plot, 
                 np.exp(log_Z_per_FE_pred[m_to_plot , ns_to_plot[0]:ns_to_plot[-1] + 1] - 
                        ratio * (m_to_plot + 1) * 
                        log_Z_per_FE_pred[0, ns_to_plot[0]:ns_to_plot[-1] + 1]), 
                 #label='m = ' + str(m_to_plot) + '; pred', 
                 linestyle=':', color=color)
    if e == 0:
        axs[e].legend(bbox_to_anchor=(1.05, 1))
    axs[e].set_xlabel(r'$n$')
    axs[e].set_yscale('log')
    if ratio == 1:
        axs[e].set_ylabel(r'$Z_m / Z_1^m$')
        axs[e].set_ylim([1e-1, 1e61])
        axs[e].set_yticks([1e0, 1e20, 1e40, 1e60])
    else:
        axs[e].set_ylabel(r'$Z_m$')
        axs[e].set_ylim([1e-1, 1e301])
        axs[e].set_yticks([1e0, 1e100, 1e200, 1e300])#, 1e300])
    axs[e].set_xticks([5, 10, 15, 20])
for ax in axs:
    ax.label_outer()
# plt.tight_layout()
plt.show()


for e, conc in enumerate(concs_to_try):
    equilibrium_multimer_concentrations = get_eq_multimer_concs(
            log_Z_per_FE_pred[:max_m_to_use, :max_n_to_use], conc)
    
    # Compute what fraction of the RNA is in each phase
    frac_monomer_per_conc[e, :] = equilibrium_multimer_concentrations[0, :] / conc
    frac_oligomer_per_conc[e, :] = np.sum(np.transpose(equilibrium_multimer_concentrations[1:max_m_for_oligomer,:]) * 
                        np.arange(2, max_m_for_oligomer + 1), 1) / conc
    frac_droplet_per_conc[e, :] = np.sum(np.transpose(equilibrium_multimer_concentrations[max_m_for_oligomer:,:]) * 
                        np.arange(max_m_for_oligomer + 1, max_m_to_use + 1), 1) / conc


plot_monomers_oligomers_droplets_frac(
    frac_monomer_per_conc[:, 2:], frac_oligomer_per_conc[:, 2:], frac_droplet_per_conc[:, 2:], 
    title = 'Prediction', 
    yticks=[0, 5, 10, 15], 
    ytick_labels=[max_n_to_use - 1, max_n_to_use-6, max_n_to_use-11, max_n_to_use-16], 
    xticks=[0, (num_concs - 1) // 2, num_concs - 1],
    xtick_labels=[r'$10$', r'$10^2$', r'$10^3$'], )



# Put monomers and dimers in by hand -- this maintains the monomers/oligomers picture very well
log_Z_per_FE_pred_md = copy.copy(log_Z_per_FE_pred)
log_Z_per_FE_pred_md[0, :] = corrected_log_Z_per_FE_dict[len_linker][FE_index, 0, :max_n_to_use]
log_Z_per_FE_pred_md[1, :] = corrected_log_Z_per_FE_dict[len_linker][FE_index, 1, :max_n_to_use]


for e, conc in enumerate(concs_to_try):
    equilibrium_multimer_concentrations = get_eq_multimer_concs(
            log_Z_per_FE_pred_md[:max_m_to_use, :max_n_to_use], conc)
    
    # Compute what fraction of the RNA is in each phase
    frac_monomer_per_conc[e, :] = equilibrium_multimer_concentrations[0, :] / conc
    frac_oligomer_per_conc[e, :] = np.sum(np.transpose(equilibrium_multimer_concentrations[1:max_m_for_oligomer,:]) * 
                        np.arange(2, max_m_for_oligomer + 1), 1) / conc
    frac_droplet_per_conc[e, :] = np.sum(np.transpose(equilibrium_multimer_concentrations[max_m_for_oligomer:,:]) * 
                        np.arange(max_m_for_oligomer + 1, max_m_to_use + 1), 1) / conc


plot_monomers_oligomers_droplets_frac(
    frac_monomer_per_conc[:, 2:], frac_oligomer_per_conc[:, 2:], frac_droplet_per_conc[:, 2:], 
    title = 'Prediction except monomers & dimers', 
    yticks=[0, 5, 10, 15], 
    ytick_labels=[max_n_to_use - 1, max_n_to_use-6, max_n_to_use-11, max_n_to_use-16], 
    xticks=[0, (num_concs - 1) // 2, num_concs - 1],
    xtick_labels=[r'$10$', r'$10^2$', r'$10^3$'], )


# =============================================================================
# Predicting further
# =============================================================================
max_n_to_use = 61
max_m_to_use = 64
max_m_for_oligomer = 4
FE_to_plot = -10
len_linker = 1  # the same thing with len_linker=4 leads to lots of large oligomers or small droplets. Not really the right thing. 
# For len_linker=1, oligomers really is basically dimers. Changing max_m_for_oligomer to 2 or to 10 doesn't have a big effect
# For len_linker=4, it really does.
FE_index = np.where(np.array(FE_from_one_bp_list_dict[len_linker])==FE_to_plot)[0][0]

num_concs = 101
concs_to_try = np.logspace(-5, max_c, num_concs)  # in units of M
frac_monomer_per_conc = np.zeros((num_concs, max_n_to_use))
frac_oligomer_per_conc = np.zeros((num_concs, max_n_to_use))
frac_droplet_per_conc = np.zeros((num_concs, max_n_to_use))

log_Z_per_FE_pred = np.zeros((max_m_to_use, max_n_to_use))
for m in range(max_m_to_use):
    for n in range(2, max_n_to_use):
        # if m < 2:  # leads to same results, but a somewhat less principled approach
        #     log_Z_per_FE_pred[m, n] = analytical_simp_predicted_logZ_all_bonds(
        #                           n, m, 4.3, len_linker >= 3, 
        #                           FE_to_plot, DNA=False, alpha_0=0.42, alpha_n=0)        
        # else:
        #     log_Z_per_FE_pred[m, n] = analytical_simp_predicted_logZ_some_bonds(
        #                           n, m, 4.3, len_linker >= 3, 
        #                           FE_to_plot, DNA=False, alpha_0=0.42, alpha_n=0, include_saddlepoint=True)
        
        loop_len_pred_multimer = 4.3 + 2.7*(len_linker==4)

        if (n==2) or (n - 0.5 - (n % 2) < 2 * NpStarFxn(
                n, 0, loop_len_pred_multimer, len_linker >= 3, 
                FE_from_one_bp, alpha_0=0.42, alpha_n=0)):
            log_Z_per_FE_pred[m, n] = analytical_simp_predicted_logZ_all_bonds(
                                  n, m, loop_len_pred_multimer, len_linker >= 3, 
                                  FE_to_plot, DNA=False, alpha_0=0.42, alpha_n=0)        
      #  # elif FE_from_one_bp < (2 * np.log((n_to_plot -6) / 3) +  # when NbStar = 3 for monomers
      #  #       T * closedNet0EntropyFxn(loop_len_pred_multimer, g)):
        else:
            log_Z_per_FE_pred[m, n] = analytical_simp_predicted_logZ_some_bonds(
                                  n, m, loop_len_pred_multimer, len_linker >= 3, 
                                  FE_to_plot, DNA=False, alpha_0=0.42, alpha_n=0, include_saddlepoint=True)
      #  # else:
      #  #     if m < 15 and n < ns_loaded_per_FE_dict[len_linker][FE_from_one_bp][m]:
      #  #         true_val = corrected_log_Z_per_FE_dict[len_linker][FE_index, m, n]
      #  #         if not np.isnan(true_val):
      #  #             log_Z_per_FE_pred[m, n] = true_val


log_Z_per_FE_pred[0, :] = corrected_log_Z_per_FE_dict[len_linker][FE_index, 0, :max_n_to_use]
log_Z_per_FE_pred[1, :] = corrected_log_Z_per_FE_dict[len_linker][FE_index, 1, :max_n_to_use]
# for m in range(2, 11):  # Doesn't change anything and just causes some errors
#     n_loaded_m = ns_loaded_per_FE_dict[len_linker][FE_to_plot][m]
#     log_Z_per_FE_pred[m, :min(max_n_to_use, n_loaded_m)] = corrected_log_Z_per_FE_dict[len_linker][FE_index, 0, :n_loaded_m]


for e, conc in enumerate(concs_to_try):
    equilibrium_multimer_concentrations = get_eq_multimer_concs(
            log_Z_per_FE_pred[:max_m_to_use, :max_n_to_use], conc)
    
    # Compute what fraction of the RNA is in each phase
    frac_monomer_per_conc[e, :] = equilibrium_multimer_concentrations[0, :] / conc
    frac_oligomer_per_conc[e, :] = np.sum(np.transpose(equilibrium_multimer_concentrations[1:max_m_for_oligomer,:]) * 
                        np.arange(2, max_m_for_oligomer + 1), 1) / conc
    frac_droplet_per_conc[e, :] = np.sum(np.transpose(equilibrium_multimer_concentrations[max_m_for_oligomer:,:]) * 
                        np.arange(max_m_for_oligomer + 1, max_m_to_use + 1), 1) / conc




plot_monomers_oligomers_droplets_frac(
    frac_monomer_per_conc[:, 2:], frac_oligomer_per_conc[:, 2:], frac_droplet_per_conc[:, 2:], 
    title = '', 
    #yticks=[2, 18, 29], ytick_labels=[max_n_to_use - 3, max_n_to_use-19, max_n_to_use - 30],
    yticks=[0, 20, 40], 
    ytick_labels=[max_n_to_use - 1,max_n_to_use - 21, max_n_to_use - 41],
    xticks=[0, (num_concs - 1) / 2, num_concs - 1],
    xtick_labels=[r'$10^1$', r'$10^2$', r'$10^3$'], 
    plot_thirumalai=True, concs_to_try=concs_to_try)


# =============================================================================
# Plot concentrations as function of m
# =============================================================================

len_linker = 4
FEs_to_plot = FE_from_one_bp_list_dict[len_linker]
num_FEs = len(FEs_to_plot)
max_m_for_oligomer = 4

n_to_plot, conc, max_m_to_plot, num_FEs_to_exclude = [8,0.1, 15, 1] # [19, 0.002, 9, 3] #[18, 0.002, 6, 2] #[8, 0.006, 15, 1]  #
# Need higher conc since we're using RNA here; Very negative FEs have NANs for results

equilibrium_multimer_concentrations_per_FE = np.zeros((
    num_FEs, max_m_to_plot, n_to_plot + 1))

for e, FE_from_one_bp in enumerate(FEs_to_plot[num_FEs_to_exclude:]):
    equilibrium_multimer_concentrations_per_FE[e + num_FEs_to_exclude, :, :] = get_eq_multimer_concs(
        corrected_log_Z_per_FE_dict[len_linker][e + num_FEs_to_exclude, :max_m_to_plot, :n_to_plot + 1], 
        conc, max_num_guesses=100)
for e in range(num_FEs_to_exclude):
    equilibrium_multimer_concentrations_per_FE[e, :, :] = equilibrium_multimer_concentrations_per_FE[num_FEs_to_exclude, :, :]


plt.figure()
for e, FE_from_one_bp in enumerate(FEs_to_plot):
    if FE_from_one_bp in [-11]: # [-15, -13, -11, -9, -7, -5]:
        plt.plot(range(1, max_m_to_plot + 1), 
                 equilibrium_multimer_concentrations_per_FE[e, :max_m_to_plot, n_to_plot] / 
                 equilibrium_multimer_concentrations_per_FE[e, 0, n_to_plot], 
                 label='Aggregation', linewidth=3, color='#1E88E5')
    if FE_from_one_bp in [-1]: # [-15, -13, -11, -9, -7, -5]:
        plt.plot(range(1, max_m_to_plot + 1), 
                 equilibrium_multimer_concentrations_per_FE[e, :max_m_to_plot, n_to_plot] / 
                 equilibrium_multimer_concentrations_per_FE[e, 0, n_to_plot], 
                 label='No aggregation', linewidth=3, color='#FFC107')
plt.ylabel('Concentration \n relative to monomer')
plt.xlabel('Multimer size ' + r'$m$')
#plt.title('n=' + str(n_to_plot) + '; conc=' + str(conc*1e3) + ' mM') # ' max_m=' + str(max_m_to_plot))
plt.legend(loc='best', fontsize=19)
plt.yscale('log')
plt.ylim([0.8e-19, 2.1])
plt.yticks([1e-15, 1e-10, 1e-5, 1])
plt.show()


# =============================================================================
# Plot the reentrant phase transition
# =============================================================================
max_m_for_oligomer = 4

# len_linker, n_to_plot, conc, max_m_to_plot, num_FEs_to_exclude = [4, 8, 0.008, 15, 1] # [19, 0.002, 9, 3] #[18, 0.002, 6, 2] #[8, 0.006, 15, 1]  #
len_linker, n_to_plot, conc, max_m_to_plot, num_FEs_to_exclude = [1, 8, 0.004, 15, 1] # [19, 0.002, 9, 3] #[18, 0.002, 6, 2] #[8, 0.006, 15, 1]  #

FEs_to_plot = FE_from_one_bp_list_dict[len_linker]
num_FEs = len(FEs_to_plot)

# Need higher conc since we're using RNA here; Very negative FEs have NANs for results

# equilibrium_multimer_concentrations_per_FE = np.zeros((num_FEs, max_m_to_plot))
# for FE_index in range(num_FEs):
#     # In order to use the function we made before, need to compute for all n's up to this n
#     equilibrium_multimer_concentrations_per_FE[FE_index] = get_eq_multimer_concs(
#         corrected_log_Z_per_FE_dict[len_linker][FE_index, :, :n_to_plot + 1], conc)[:, -1]

equilibrium_multimer_concentrations_per_FE = np.zeros((
    num_FEs, max_m_to_plot, n_to_plot + 1))

for e, FE_from_one_bp in enumerate(FEs_to_plot[num_FEs_to_exclude:]):
    equilibrium_multimer_concentrations_per_FE[e + num_FEs_to_exclude, :, :] = get_eq_multimer_concs(
        corrected_log_Z_per_FE_dict[len_linker][e + num_FEs_to_exclude, :max_m_to_plot, :n_to_plot + 1], 
        conc, max_num_guesses=100)
for e in range(num_FEs_to_exclude):
    equilibrium_multimer_concentrations_per_FE[e, :, :] = equilibrium_multimer_concentrations_per_FE[num_FEs_to_exclude, :, :]


seqs_to_plot = ['CAAUUC', 'AGCGCA', 'AGCAUGCA']  # 'ACCGGA', 'AGCA', 
seqs_to_plot_Fb_RNA = []
seqs_to_plot_Fb_DNA = []
for short_repeat in seqs_to_plot:
    for DNA in [True, False]:
        if DNA:
            short_repeat = short_repeat.replace('U', 'T')
        else:
            short_repeat = short_repeat.replace('T', 'U')
        short_repeat_landscape = LandscapeFold(
            [short_repeat, short_repeat], 
            DNA=[DNA, DNA],
            makeFigures=False, 
            printProgressUpdate=False,
            T=273.15 + 37, 
            minBPInStem=len(short_repeat) - 2,
            onlyConsiderBondedStrands=True,
            ).mainLandscapeCalculation()
        if DNA:
            seqs_to_plot_Fb_DNA += [short_repeat_landscape.allBondEnergies[0] - 
                            (273.15 + 37) * short_repeat_landscape.allBondEntropies[0]]
        else:
            seqs_to_plot_Fb_RNA += [short_repeat_landscape.allBondEnergies[0] - 
                            (273.15 + 37) * short_repeat_landscape.allBondEntropies[0]]


plt.figure()
plt.plot(FEs_to_plot, 
         equilibrium_multimer_concentrations_per_FE[:, 0, n_to_plot] / conc, 
         label='monomers', linewidth=3,
         color='#D81B60')
plt.plot(FEs_to_plot, 
         np.sum(
             [equilibrium_multimer_concentrations_per_FE[:, m, n_to_plot] * (m+1) / conc
              for m in range(1, max_m_for_oligomer)], 0), #'.',
         label='oligomers', linewidth=3,
         color='#FFC107'
         )
plt.plot(FEs_to_plot, 
         np.sum(
             [equilibrium_multimer_concentrations_per_FE[:, m, n_to_plot] * (m+1) / conc
              for m in range(max_m_for_oligomer, max_m_to_plot)], 0), 
         label='aggregates', linewidth=3,
         color='#1E88E5')        
plt.ylabel('Equilibrium fraction')
plt.xlabel(r'$F_b$ (kcal/mol)')
#plt.yscale('log')
# plt.title('n=' + str(n_to_plot) + '; conc=' + str(conc*1e3) + ' mM') # ' max_m=' + str(max_m_to_plot))
plt.legend(loc='best', fontsize=13)
#plt.xlim([2,15])
plt.ylim([-0.03, 1.03])
plt.yticks([0, 0.5, 1])
ax = plt.gca()
secax = ax.secondary_xaxis('top')
secax.set_xticks(seqs_to_plot_Fb_DNA + seqs_to_plot_Fb_RNA)
secax.set_xticklabels([])

for e, short_repeat_seq in enumerate(seqs_to_plot):
    for e_d, DNA in enumerate([True, False]):
        plt.text([seqs_to_plot_Fb_DNA, seqs_to_plot_Fb_RNA][e_d][e] - 0.5, 
                 #-0.38 + (6-len(short_repeat_seq))/15, 
                 1.055,
                 s=['d', 'r'][e_d] + short_repeat_seq[1:].replace(['U', 'T'][e_d], ['T', 'U'][e_d]), 
                 rotation=50, 
                 #alpha=0.2
                 )
plt.show()



# =============================================================================
# Confirm that this isn't just because of low max_m -- use the extrapolated data
# =============================================================================
max_m_to_use = 64
max_n_to_use = 21
log_Z_per_FE_pred = np.zeros((num_FEs, max_m_to_use, max_n_to_use))
for m_to_plot in range(max_m_to_use):
    for n_to_plot in range(3, max_n_to_use):  # n=2 gives issues
        for e, FE_from_one_bp in enumerate(FEs_to_plot):
            Np_pred_monomer = NpStarFxn(n_to_plot, 0, 4, len_linker >= 3, FE_from_one_bp, alpha_0=0.42, alpha_n=0)
            Np_pred_multimer = NpStarFxn(n_to_plot, m_to_plot, 4, len_linker >= 3, FE_from_one_bp, alpha_0=0.42, alpha_n=0)
            
            loop_len_pred_monomer = min(
                len_linker * (n_to_plot-2), 
                (4.3 + 2.7*(len_linker==4)) * n_to_plot / (2 * (Np_pred_monomer)))
            loop_len_pred_multimer = min(
                len_linker * (n_to_plot-2), 
                (4.3 + 2.7*(len_linker==4)) * n_to_plot *(m_to_plot + 1) / (2 * (Np_pred_multimer)))

            # if (n==2) or (FE_from_one_bp < (2 * np.log(4 / (n_to_plot-2)) +   # when NbStar = n/2 - 1 for monomers
            #       T * closedNet0EntropyFxn(loop_len_pred_monomer, g)) - 1):
            if (n==2) or (n_to_plot - 0.5 - (n_to_plot % 2) < 2 * NpStarFxn(
                    n_to_plot, 0, loop_len_pred_multimer, len_linker >= 3, 
                    FE_from_one_bp, alpha_0=0.42, alpha_n=0)):
                log_Z_per_FE_pred[e, m_to_plot, n_to_plot] = analytical_simp_predicted_logZ_all_bonds(
                                      n_to_plot, m_to_plot, loop_len_pred_multimer, len_linker >= 3, 
                                      FE_from_one_bp, DNA=False, alpha_0=0.42, alpha_n=0)    
            elif FE_from_one_bp < (2 * np.log((n_to_plot -6) / 3) +  # when NbStar = 3 for monomers
                  T * closedNet0EntropyFxn(loop_len_pred_monomer, g)):
                log_Z_per_FE_pred[e, m_to_plot, n_to_plot] = analytical_simp_predicted_logZ_some_bonds(
                                      n_to_plot, m_to_plot, loop_len_pred_multimer, len_linker >= 3, 
                                      FE_from_one_bp, DNA=False, alpha_0=0.42, alpha_n=0, include_saddlepoint=True) 
            else:
                if m_to_plot < 15 and n_to_plot < ns_loaded_per_FE_dict[len_linker][FE_from_one_bp][m_to_plot]:
                    true_val = corrected_log_Z_per_FE_dict[len_linker][e, m_to_plot, n_to_plot]
                    if not np.isnan(true_val):
                        log_Z_per_FE_pred[e, m_to_plot, n_to_plot] = true_val


# Check the extrapolation
for ratio in [1]:
    fig = plt.figure()
    gs = fig.add_gridspec(2, 1, wspace=0.15, hspace=0.25)
    axs_full = gs.subplots(sharey='row')
    axs = axs_full.flat
    for e, n_to_plot in enumerate([18, 19]):
        for m_to_plot in [0, 1, 2, 3, 5, 7, 9]: # range(2, ns_loaded_per_FE_4[-10][m_to_plot]): # [4, 5, 6, 7, 8, 9]: #
            color = next(axs[e]._get_lines.prop_cycler)['color']
            axs[e].plot(FE_from_one_bp_list_dict[len_linker], 
                     np.exp(corrected_log_Z_per_FE_dict[len_linker][:, m_to_plot , n_to_plot] - 
                     ratio * (m_to_plot + 1) * corrected_log_Z_per_FE_dict[len_linker][:, 0, n_to_plot]), 
                     label=m_to_plot, color=color, )
            # color = next(axs[e]._get_lines.prop_cycler)['color']
            axs[e].plot(FEs_to_plot, 
                     np.exp(log_Z_per_FE_pred[:, m_to_plot , n_to_plot] - 
                     ratio * (m_to_plot + 1) * log_Z_per_FE_pred[:, 0, n_to_plot]), 
                     #label='m = ' + str(m_to_plot) + '; pred', 
                     linestyle=':', color=color)
    
        if e == 0:
            axs[e].legend(bbox_to_anchor=(1.05,1), title=r'$m$')
        axs[e].set_xlabel(r'$F_b$' + ' (kcal/mol)')
        if ratio == 0:
            axs[e].set_ylabel(r'$Z_m$')
            axs[e].set_yticks([1e0, 1e100, 1e200, 1e300])
            axs[e].set_ylim([1e-5, 1e301])
        if ratio == 1:
            axs[e].set_ylabel(r'$Z_m / Z_1^m$')
            axs[e].set_yticks([1e0, 1e20, 1e40, 1e60])
            axs[e].set_ylim([1e-5, 1e65])
        axs[e].set_yscale('log')
        axs[e].set_xticks([-15, -10, -5, 0])
    for ax in axs:
        ax.label_outer()
    # plt.tight_layout()
    plt.show()






n_to_plot, conc, max_m_to_plot, num_FEs_to_exclude = [18, 0.01, 64, 0] # [19, 0.002, 9, 3] #[18, 0.002, 6, 2] #[8, 0.006, 15, 1]  #
# Need higher conc since we're using RNA here; Very negative FEs have NANs for results

# equilibrium_multimer_concentrations_per_FE = np.zeros((num_FEs, max_m_to_plot))
# for FE_index in range(num_FEs):
#     # In order to use the function we made before, need to compute for all n's up to this n
#     equilibrium_multimer_concentrations_per_FE[FE_index] = get_eq_multimer_concs(
#         corrected_log_Z_per_FE_dict[len_linker][FE_index, :, :n_to_plot + 1], conc)[:, -1]

equilibrium_multimer_concentrations_per_FE = np.zeros((
    num_FEs, max_m_to_plot, n_to_plot + 1))

for e, FE_from_one_bp in enumerate(FEs_to_plot[num_FEs_to_exclude:]):
    equilibrium_multimer_concentrations_per_FE[e + num_FEs_to_exclude, :, :] = get_eq_multimer_concs(
        log_Z_per_FE_pred[e + num_FEs_to_exclude, :max_m_to_plot, :n_to_plot + 1], 
        # corrected_log_Z_per_FE_dict[len_linker][e + num_FEs_to_exclude, :max_m_to_plot, :n_to_plot + 1], 
        conc, max_num_guesses=100)
for e in range(num_FEs_to_exclude):
    equilibrium_multimer_concentrations_per_FE[e, :, :] = equilibrium_multimer_concentrations_per_FE[num_FEs_to_exclude, :, :]


plt.figure()
plt.plot(FEs_to_plot, 
         equilibrium_multimer_concentrations_per_FE[:, 0, n_to_plot] / conc, 
         label='monomers', linestyle='--')
plt.plot(FEs_to_plot, 
         np.sum(
             [equilibrium_multimer_concentrations_per_FE[:, m, n_to_plot] * (m+1) / conc
              for m in range(1, 10)], 0), #'.',
         label='oligomers', linestyle=':'
         )
plt.plot(FEs_to_plot, 
         np.sum(
             [equilibrium_multimer_concentrations_per_FE[:, m, n_to_plot] * (m+1) / conc
              for m in range(10, max_m_to_plot - 10)], 0), #'.',
         label='large multimers', linestyle=':'
         )
plt.plot(FEs_to_plot, 
         np.sum(
             [equilibrium_multimer_concentrations_per_FE[:, m, n_to_plot] * (m+1) / conc
              for m in range(max_m_to_plot - 10, max_m_to_plot)], 0), 
         label='aggregates')        
plt.ylabel('Equilibrium fraction')
plt.xlabel(r'$F_b$')
#plt.yscale('log')
#plt.title('n = ' + str(n_to_plot) + '; conc = ' + str(conc))
plt.legend(loc='best', fontsize=15)
#plt.xlim([2,15])
plt.ylim([-0.03, 1.03])
plt.yticks([0, 0.5, 1])
plt.show()






# =============================================================================
# Make phase diagram
# =============================================================================

def ctot_thresh(n, betaF, beta_DeltaF, adjacent_stickers_can_bind, 
                alpha_0=0.42, alpha_n=0, useMstar2=False, include_dimer_term=True):
    # betaF = beta_eff_Fb
    if adjacent_stickers_can_bind:
        if n - (n%2) - 0.5 > 2 * n / (2 + np.exp(betaF/2)):
            ctot_pred = (np.pi / (12 * n**2)) * np.sqrt(1 + 2 * np.exp(-betaF/2)) * (2 + np.exp(betaF/2))**2 
        else:
            mStar = 2 * (1 + 2 * np.exp(-betaF/2)) / n
            x = (1 + np.exp(betaF/2)/2)**(-n)
            if n % 2 == 0:
                ctot_pred = np.sqrt(1 + 2 * np.exp(-betaF/2)) * (2 + np.exp(betaF/2))**2 / (2 * np.pi * n**2) * polygamma(1, mStar)
                ctot_pred += np.sqrt(8 / (np.pi * n**3)) * (
                    polylog(3/2, x) - x**(1+mStar) * lerchphi(x, 3/2, 1 + mStar)
                    )
            else:
                def so(z, m):
                    return(z/np.sqrt(2) * (lerchphi(z**2, 1/2, 1/2) - z**(m + 1) * lerchphi(z**2, 1/2, m/2 + 1)))
                def se(z, m):
                    return(2**(-3/2) * (polylog(3/2, z**2) - z**(m + 2) * lerchphi(z**2, 3/2, m/2 + 1)))
                ctot_pred = np.sqrt(1 + 2 * np.exp(-betaF/2)) * (2 + np.exp(betaF/2))**2 / (2 * np.pi * n**2) * polygamma(1, mStar + 1)
                ctot_pred += np.exp(betaF/2) * np.sqrt(2/(np.pi * n)) * so(x, mStar)
                ctot_pred += np.sqrt(8/(np.pi * n**3)) * se(x, mStar)
    else:
        alpha = alpha_0 + n * alpha_n  #n/50 + 0.25 + beta_eff_Fb / 50 # 0.42
        def fFxn(x):
            # x = betaF
            return(1 - (np.exp(x / 4) / np.sqrt(4 + np.exp(x / 2))))
        def fOfFFxn(x):
            f = fFxn(x)
            logFofF = f * (np.log(2) - x/2 + np.log(1 - f) - np.log(1 - f/2) + np.log(1/f - 1)) + 2 * (np.log(1 - f/2) - np.log(1 - f))
            return(np.exp(logFofF))
        if n - 4 > n * fFxn(betaF): #n - 3 - 1.5 > n * fFxn(betaF): #n - 1 - (1-(n%2)) - 1.5 > n * fFxn(betaF)  #n - 8 > n * fFxn(betaF)
            return(decimal.Decimal(
                (np.pi / (3 * ((n + alpha) * fFxn(betaF))**2 * ((fOfFFxn(betaF))**alpha))) * 
                np.exp(beta_DeltaF)))
        else:
            if useMstar2:
                if n % 2 == 0:  # i.e. even n
                    Z1 = (n * (n + 2) / 8) * np.exp(-betaF * (n - 2) / 2)
                else:
                    Z1 = np.exp(-betaF * (n - 1) / 2)
                if include_dimer_term:
                    dimer_term = np.exp(
                        decimal.Decimal(-2 * n - alpha) * decimal.Decimal(fOfFFxn(betaF)).ln() -
                        decimal.Decimal(betaF * (n + alpha/2)))
                else:
                    dimer_term = decimal.Decimal(0)
                return(((decimal.Decimal(np.pi / 3 - 2 / np.pi) / decimal.Decimal(((n + alpha) * fFxn(betaF))**2) + 
                       decimal.Decimal(fOfFFxn(betaF)**(-n) * Z1) +
                       dimer_term) / decimal.Decimal(fOfFFxn(betaF)**alpha)) * decimal.Decimal(np.exp(beta_DeltaF)))
            else:
                mStar = (2) / (n * (1 - fFxn(betaF)))
                sum1arg = (fOfFFxn(betaF) * np.exp(betaF / 2))**(-n - alpha)

                if n % 2 == 0:  # i.e. even n
                    ctot_pred = (np.exp(betaF * alpha / 2) * sum1arg * (1 - sum1arg**mStar) / (1 - sum1arg) + 
                                 2 * fOfFFxn(betaF)**(-alpha) * polygamma(1, mStar + 1) / (
                                     np.pi * ((n + alpha) * fFxn(betaF))**2))
                else:
                    ctot_pred = (np.exp(betaF * alpha / 2) * sum1arg * (
                        1 + np.exp(betaF / 2) - sum1arg**(mStar / 2) * (1 + sum1arg**(1/2) * np.exp(betaF/2)))
                        / (1 - sum1arg)) + (
                             2 * fOfFFxn(betaF)**(-alpha) * polygamma(1, mStar + 1) / (
                                 np.pi * ((n + alpha) * fFxn(betaF))**2))
                return(decimal.Decimal(ctot_pred * np.exp(beta_DeltaF)))
    return(ctot_pred * np.exp(beta_DeltaF))


ns_to_plot = range(2, 61)
numF = 101
betaF_to_plot = np.linspace(-20, 5, num=numF)
ctot_thresh_withNeigh = np.zeros((len(ns_to_plot), len(betaF_to_plot)))
ctot_thresh_noNeigh = np.zeros((len(ns_to_plot), len(betaF_to_plot)), dtype=decimal.Decimal)
ctot_thresh_noNeigh_log = np.zeros((len(ns_to_plot), len(betaF_to_plot)))
for e1, n in enumerate(ns_to_plot):
    for e2, betaF in enumerate(betaF_to_plot):
        ctot_thresh_withNeigh[e1, e2] = ctot_thresh(n, betaF, 0, True)
        ctot_thresh_noNeigh[e1, e2] = ctot_thresh(n, betaF, 0, False, alpha_0=0.42, alpha_n=0)
        ctot_thresh_noNeigh_log[e1, e2] = float(ctot_thresh_noNeigh[e1, e2].ln())



plt.figure()
# plt.plot(betaF_to_plot, ctot_thresh_withNeigh[5, :], label='n=7', linestyle='--')
# plt.plot(betaF_to_plot, ctot_thresh_withNeigh[6, :], label='n=8')
plt.plot(betaF_to_plot, ctot_thresh_withNeigh[35, :], label='n=37', linestyle='--', color='b')
plt.plot(betaF_to_plot, ctot_thresh_withNeigh[36, :], label='n=38', color='b')
# plt.plot(betaF_to_plot, ctot_thresh_noNeigh[35, :], label='n=37', linestyle='--', color='orange')
# plt.plot(betaF_to_plot, ctot_thresh_noNeigh[36, :], label='n=38', color='orange')

# plt.scatter(betaF_to_plot, ctot_thresh_withNeigh[35, :], label='n=37', c=ctot_thresh_withNeigh[35, :], cmap='turbo')
# plt.scatter(betaF_to_plot, ctot_thresh_withNeigh[36, :], label='n=38', c=ctot_thresh_withNeigh[35, :], cmap='turbo')

plt.xlabel(r'$\beta F$')
plt.ylabel(r'$c^{tot}_{thresh}/e^{\beta \Delta F}$')
plt.legend()
# plt.yscale('log')
plt.show()




plt.figure()
# plt.plot(betaF_to_plot, ctot_thresh_withNeigh[5, :], label='n=7', linestyle='--', color='#D81B60', linewidth=3.)
# plt.plot(betaF_to_plot, ctot_thresh_withNeigh[6, :], label='n=8', color='#D81B60', linewidth=3.)

# plt.plot(betaF_to_plot, ctot_thresh_withNeigh[15, :], label='n=17', linestyle='--', color='#D81B60', linewidth=3.)
# plt.plot(betaF_to_plot, ctot_thresh_withNeigh[16, :], label='n=18', color='#D81B60', linewidth=3.)

plt.plot(betaF_to_plot, ctot_thresh_withNeigh[25, :], label='27', linestyle='--', color='#D81B60', linewidth=3.)
plt.plot(betaF_to_plot, ctot_thresh_withNeigh[26, :], label='28', color='#D81B60', linewidth=3.)

plt.plot(betaF_to_plot, ctot_thresh_withNeigh[35, :], label='37', linestyle='--', color='#FFC107', linewidth=3.)
plt.plot(betaF_to_plot, ctot_thresh_withNeigh[36, :], label='38', color='#FFC107', linewidth=3.)

plt.plot(betaF_to_plot, ctot_thresh_withNeigh[45, :], label='47', linestyle='--', color='#1E88E5', linewidth=3.)
plt.plot(betaF_to_plot, ctot_thresh_withNeigh[46, :], label='48', color='#1E88E5', linewidth=3.)

plt.xlabel(r'$\beta F$')
plt.ylabel(r'$c^{tot}_{thresh}/e^{\beta \Delta F}$')
plt.yscale('log')
plt.yticks([1e-1, 1e-2, 1e-3])
plt.legend(bbox_to_anchor=(1.05, 1), title=r'$n$')
plt.show()




plt.figure()
plt.plot(betaF_to_plot, ctot_thresh_noNeigh[25, :], label='27', linestyle='--', color='#D81B60', linewidth=3.)
plt.plot(betaF_to_plot, ctot_thresh_noNeigh[26, :], label='28', color='#D81B60', linewidth=3.)

plt.plot(betaF_to_plot, ctot_thresh_noNeigh[35, :], label='37', linestyle='--', color='#FFC107', linewidth=3.)
plt.plot(betaF_to_plot, ctot_thresh_noNeigh[36, :], label='38', color='#FFC107', linewidth=3.)

plt.plot(betaF_to_plot, ctot_thresh_noNeigh[45, :], label='47', linestyle='--', color='#1E88E5', linewidth=3.)
plt.plot(betaF_to_plot, ctot_thresh_noNeigh[46, :], label='48', color='#1E88E5', linewidth=3.)

plt.xlabel(r'$\beta F$')
plt.ylabel(r'$c^{tot}_{thresh}/e^{\beta \Delta F}$')
plt.yscale('log')
plt.yticks([1e-1, 1e-2, 1e-3, 1e-4])
plt.legend(bbox_to_anchor=(1.05, 1), title=r'$n$')
plt.show()




plt.figure()
ax1 = plt.gca()
im1 = ax1.imshow(np.log(np.flipud(ctot_thresh_withNeigh)), #np.flipud(np.transpose(frac_monomer_per_conc)), 
                 aspect='auto', cmap='turbo')
cb = plt.colorbar(im1, ax=ax1, label=r'$c^{tot}_{thresh}/e^{\beta \Delta F}$')
cb.set_ticks([np.log(10), 0, 
              -np.log(10), -np.log(100), 
              -np.log(1000)])
cb.set_ticklabels([r'$10^{1}$', r'$10^{0}$', #r'$\frac{1}{2}$', 
                   r'$10^{-1}$', r'$10^{-2}$',
                   r'$10^{-3}$'])

ax1.set_xticks([0, (numF-1)/5 * 2, (numF-1)/5 * 4])
ax1.set_xticklabels([-20, -10, 0])
ax1.set_yticks([0, 20, 40])
ax1.set_yticklabels([ns_to_plot[-1], ns_to_plot[-21], ns_to_plot[-41]])
plt.xlabel(r'$\beta F$')
plt.ylabel(r'$n$')
plt.title('Allowing neighbor binding')
plt.show()




plt.figure()
ax1 = plt.gca()
im1 = ax1.imshow(np.flipud(ctot_thresh_noNeigh_log),
                 aspect='auto', cmap='turbo')
cb = plt.colorbar(im1, ax=ax1, label=r'$c^{tot}_{thresh}/e^{\beta \Delta F}$')
cb.set_ticks([0, -np.log(100), -np.log(1e4)])
cb.set_ticklabels([r'$10^{0}$', r'$10^{-2}$', r'$10^{-4}$'])

ax1.set_xticks([0, (numF-1)/5 * 2, (numF-1)/5 * 4])
ax1.set_xticklabels([-20, -10, 0])
ax1.set_yticks([0, 20, 40])
ax1.set_yticklabels([ns_to_plot[-1], ns_to_plot[-21], ns_to_plot[-41]])
plt.xlabel(r'$\beta F$')
plt.ylabel(r'$n$')
plt.title('Disallowing neighbor binding')
plt.show()





# Combining the plots
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9.75, 3))
ax1, ax2 = axs
im1 = ax1.imshow(np.flipud(ctot_thresh_noNeigh_log),
                 aspect='auto', cmap='turbo',   # cubehelix_r
                 vmin=np.min(ctot_thresh_noNeigh_log),
                 vmax=np.max(np.log(ctot_thresh_withNeigh)))
ax1.set_xticks([0, (numF-1)/5 * 2, (numF-1)/5 * 4])
ax1.set_xticklabels([-20, -10, 0])
ax1.set_yticks([0, 20, 40])
ax1.set_yticklabels([ns_to_plot[-1], ns_to_plot[-21], ns_to_plot[-41]])
ax1.set_xlabel(r'$\beta F$')
ax1.set_ylabel(r'$n$')
ax1.set_title('Disallowing \n neighbor binding')

im2 = ax2.imshow(np.log(np.flipud(ctot_thresh_withNeigh)),
                 aspect='auto', cmap='turbo', 
                 vmin=np.min(ctot_thresh_noNeigh_log),
                 vmax=np.max(np.log(ctot_thresh_withNeigh)))
ax2.set_xticks([0, (numF-1)/5 * 2, (numF-1)/5 * 4])
ax2.set_xticklabels([-20, -10, 0])
ax2.set_yticks([0, 20, 40])
ax2.set_yticklabels([ns_to_plot[-1], ns_to_plot[-21], ns_to_plot[-41]])
ax2.set_xlabel(r'$\beta F$')
ax2.set_title('Allowing \n neighbor binding')


# Colorbar
cb = fig.colorbar(im1, ax=axs, label=r'$c^{tot}_{thresh}/e^{\beta \Delta F}$')
cb.set_ticks([0, -np.log(100), -np.log(1e4)])
cb.set_ticklabels([r'$10^{0}$', r'$10^{-2}$', r'$10^{-4}$'])

plt.show()




# =============================================================================
# Draw random RNA
# =============================================================================
persistence = 0.8
len_RNA = 1000

np.random.seed(3)
random_x = [0]
random_y = [0]
curr_v = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
for _ in range(len_RNA):
    curr_x, curr_y = (random_x[-1], random_y[-1])
    angle = 2 * np.pi * np.random.random()
    v_perturb = np.array([np.cos(angle), np.sin(angle)])
    curr_v = persistence * curr_v + (1 - persistence) * v_perturb
    curr_v /= np.linalg.norm(curr_v)
    random_x += [curr_x + curr_v[0]]
    random_y += [curr_y + curr_v[1]]
    
num_stickers = 10
len_sticker = int(len_RNA / num_stickers)
frac_sticker = 1/3
plt.figure()
for i in range(num_stickers):
    plt.plot(random_x[i * len_sticker : i * len_sticker + int(len_sticker * frac_sticker)],
             random_y[i * len_sticker : i * len_sticker + int(len_sticker * frac_sticker)],
             color='#1E88E5', linewidth=5)
    plt.plot(random_x[i * len_sticker + int(len_sticker * frac_sticker) : (i + 1) * len_sticker],
             random_y[i * len_sticker + int(len_sticker * frac_sticker) : (i + 1) * len_sticker],
             color='#FFC107', linewidth=5)
    plt.text(-35, -10, "3'", fontsize=28)
    plt.text(random_x[-1] - 30, random_y[-1] -5, "5'", fontsize=28)
plt.axis('off')
plt.show()


# =============================================================================
# forna code for Fig 1
# =============================================================================
# >noneigh
# AGCAGCAGCA
# .((....)).

# >withneigh
# AGCAAAAGCA
# .((....)).

# >withneighbad
# AGCAGCA
# .((.)).


# >noneigh
# 1,4,7,10:#FFC107 
# 2-3,8-9:#098AFB
# 5-6:#A9D7FF
# >withneigh
# 1,4,5,6,7,10:#FFC107 
# 2-3,8-9:#098AFB
# >withneighbad
# 1,4,7:#FFC107 
# 2-3,5-6:#098AFB



# >GCA10int1
# GCAGCAGCAGCAGCAGCAGCAGCAGCAGCA
# ((....((.......)).((....)).)).

# >GCA10int2
# GCAGCAGCAGCAGCAGCAGCAGCAGCAGCA
# ...((.((....)).))....((....)).

# >GCA10int3
# GCAGCAGCAGCAGCAGCAGCAGCAGCAGCA
# ((.((....((....))....))....)).

# >GCA10int4
# GCAGCAGCAGCAGCAGCAGCAGCAGCAGCA
# ((....)).((....((....))....)).


# >GCA10int1
# 1-2,7-8,16-17,19-20,25-26,28-29:#098AFB 
# 3,6,9,12,15,18,21,24,27,30:#FFC107
# 4-5,10-11,13-14,22-23:#A9D7FF
# >GCA10int2
# 4-5,7-8,13-14,16-17,22-23,28-29:#098AFB 
# 3,6,9,12,15,18,21,24,27,30:#FFC107
# 1-2,10-11,19-20,25-26:#A9D7FF
# >GCA10int3
# 1-2,4-5,10-11,16-17,22-23,28-29:#098AFB 
# 3,6,9,12,15,18,21,24,27,30:#FFC107
# 7-8,13-14,19-20,25-26:#A9D7FF
# >GCA10int4
# 1-2,7-8,10-11,16-17,22-23,28-29:#098AFB 
# 3,6,9,12,15,18,21,24,27,30:#FFC107
# 4-5,13-14,19-20,25-26:#A9D7FF



# >GCA10s1
# GCAGCAGCAGCAGCAGCAGCAGCAGCAGCA
# ...((.((.((.((....)).)).)).)).

# >GCA10s2
# GCAGCAGCAGCAGCAGCAGCAGCAGCAGCA
# ((.((.((.((....)).))....)).)).


# >GCA10s1
# 4-5,7-8,10-11,13-14,19-20,22-23,25-26,28-29:#098AFB 
# 3,6,9,12,15,18,21,24,27,30:#FFC107
# 1-2,16-17:#A9D7FF
# >GCA10s2
# 1-2,4-5,7-8,10-11,16-17,19-20,25-26,28-29:#098AFB 
# 3,6,9,12,15,18,21,24,27,30:#FFC107
# 13-14,22-23:#A9D7FF



# >GCA10w1
# GCAGCAGCAGCAGCAGCAGCAGCAGCAGCA
# ......((.............)).......
# >GCA10w2
# GCAGCAGCAGCAGCAGCAGCAGCAGCAGCA
# ............((..........))....

# >GCA10w3
# GCAGCAGCAGCAGCAGCAGCAGCAGCAGCA
# ...((..........)).............


# >GCA10w1
# 7-8,22-23:#098AFB 
# 3,6,9,12,15,18,21,24,27,30:#FFC107
# 1-2,4-5,10-11,13-14,16-17,19-20,25-26,28-29:#A9D7FF
# >GCA10w2
# 13-14,25-26:#098AFB 
# 3,6,9,12,15,18,21,24,27,30:#FFC107
# 1-2,4-5,7-8,10-11,16-17,19-20,22-23,28-29:#A9D7FF
# >GCA10w3
# 4-5,16-17:#098AFB 
# 3,6,9,12,15,18,21,24,27,30:#FFC107
# 1-2,7-8,10-11,13-14,19-20,22-23,25-26,28-29:#A9D7FF



# >dimer1
# AGCAGCAGCAGCAGCAGCAGCAGCAGCAGCA
# .((.((....)).))................

# >dimer2
# AGCAGCAGCAGCAGCAGCAGCAGCAGCAGCA
# ................((.((....)).)).


# >dimer1
# 1-31:#098AFB
# 1,4,7,10,13,16,19,22,25,28,31:#FFC107
# >dimer2
# 1-31:#098AFB
# 1,4,7,10,13,16,19,22,25,28,31:#FFC107
