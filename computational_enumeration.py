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


def twoListsShareElement(a,b):
# =============================================================================
#    Check if two lists, a and b, share any items. This is supposedly the fastest way to test this.
#    From https://stackoverflow.com/questions/3170055/test-if-lists-share-any-items-in-python
#    Returns True if the lists share any elements, False otherwise
# =============================================================================
    return(not set(a).isdisjoint(b))


def flatten(l): #From StackExchange
    #Flatten a list -- given a list of sublists, concatenate the sublists into one long list
    return([item for sublist in l for item in sublist])

        
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


toSave = False
tryToLoadVariables = False

# =============================================================================
# Parameters we can vary in the experiment
# =============================================================================
T37 = 273.15 + 37
len_repeat_bp = 2  # number of nts comprising each complementary section of repeat
len_linker = 4  # number of nts comprising each non-complementary section
T = T37
FE_from_one_bp = -10 
print(FE_from_one_bp)

len_repeat = len_repeat_bp + len_linker
# if linker is long enough, adjacent stickers can bind to form a hairpin
adjacent_stickers_can_bind = len_linker >= 3

# Note: n=0 means no stickers in a strand; n=1 means one sticker, etc.

# =============================================================================
# Compute loop entropies
# =============================================================================
kB = 0.0019872

b = 0.8 / 0.33
g = 3 / (2 * b)

vs = 0.02
logVs = np.log(vs)

def closedNet0EntropyFxn(s1, g):
    ent = (3/2) * np.log(g / (np.pi * s1))
    return(kB * (ent + logVs))
    
# precompute to cut down on compute time (barely)
closedNet0EntropyList = np.array(
        [0] + [closedNet0EntropyFxn(s1, g) for s1 in range(1, 3000)])


# We can also easily correct for differences between NNDB and our loop entropy model,
# though we don't do so in this work

## Consider the NNDB special 1x1 internal loop cost for this particular loop
## 0.9 cost of the loop - (-1.5-1.5)=+3 cost for the two terminal mismatches which 
## are redundant since they are included in the cost of the loop
#closedNet0EntropyList[4] = -3.9 / T
#
## Consider the NNDB cost of the particular hairpin loop formed
#closedNet0EntropyList[5] = -(5.6) / T   # not very different from what we already have
## 5.6 cost of hairpin loop of 4 nts, -1.5 for mismatch (which we've already considered) 
## gives total cost of 4.1 in agreement with mfold.
#
## Consider the NNDB cost of the first (smallest) multiloop
#closedNet0EntropyList[6] = XXX
#
## Consider the NNDB cost of the 1st asymmetric internal loop
#closedNet0EntropyList[7] = -(2.0 + 0.6*np.abs(4-1) + 2 * 1.5) / T  
## 2 is cost of initiating internal loop of 5 nts, plus there's an asymmetric penalty
## Also, "In the case of 1×(n–1) loops, the mismatches are set to 0 kcal/mol."
## So we add back in 1.5*2 for the two terminal mismatches we considered
#
## Consider the NNDB cost of the 2nd smallest hairpin
#closedNet0EntropyList[8] = -(6.0) / T  # cost of hairpin loop of 7 nts


def loopEntropyFxn(loop_length):
    # For outerloops that have no entropy cost, it's easier to keep track of them
    # as if they were of length infinity. We then don't need to treat them as a special
    # case, since when we add 2 or 3 to their length they stay at infinity. However,
    # we need to have their entropy cost be zero.
    if loop_length == np.inf:
        return(0)
    # return(0)  # to ignore loop entropies
    return(closedNet0EntropyList[loop_length])
    

# Partition functions are too big to keep track of directly; instead we keep track of their logs
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


# =============================================================================
# Construct functions to save and load partition functions
# =============================================================================
folder_name_init = '/Users/your_folder_name_here/stored_partition_fxns/'
folder_name = folder_name_init
folder_name += 'len_repeat_bp_' + str(len_repeat_bp) + '/'
folder_name += 'len_linker_' + str(len_linker) + '/'
folder_name += 'T_' + str(T) + '/'
folder_name += 'FE_from_one_bp_' + str(FE_from_one_bp) + '/'
folder_name += 'b_' + str(b) + '/'
folder_name += 'vs_' + str(vs) + '/'

if toSave or tryToLoadVariables:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def saveFxn(multimers_stored_log_boltzmann_factor, multimers_stored_log_boltzmann_factor_with_outerloop,
            multimers_stored_log_boltzmann_factor_per_outerloop, ns_loaded):
    
    save(multimers_stored_log_boltzmann_factor, folder_name + 'multimers_stored_log_boltzmann_factor')
    save(multimers_stored_log_boltzmann_factor_with_outerloop, folder_name + 'multimers_stored_log_boltzmann_factor_with_outerloop')
    save(multimers_stored_log_boltzmann_factor_per_outerloop, folder_name + 'multimers_stored_log_boltzmann_factor_per_outerloop')
    save(ns_loaded, folder_name + 'ns_loaded')
    

def loadFxn(max_m):
    start = time.time()
    try:
        multimers_stored_log_boltzmann_factor = load(folder_name + 'multimers_stored_log_boltzmann_factor')
        multimers_stored_log_boltzmann_factor_with_outerloop = load(folder_name + 'multimers_stored_log_boltzmann_factor_with_outerloop')
        multimers_stored_log_boltzmann_factor_per_outerloop = load(folder_name + 'multimers_stored_log_boltzmann_factor_per_outerloop')
        ns_loaded = load(folder_name + 'ns_loaded')
        if len(ns_loaded) < max_m:
            ns_loaded += [0] * (max_m - len(ns_loaded))
    except:
        multimers_stored_log_boltzmann_factor = dict()
        multimers_stored_log_boltzmann_factor_with_outerloop = dict()
        multimers_stored_log_boltzmann_factor_per_outerloop = dict()
        ns_loaded = [0] * max_m
        
    print('Time to load variables = ' + str(time.time() - start))
    return(multimers_stored_log_boltzmann_factor, multimers_stored_log_boltzmann_factor_with_outerloop,
           multimers_stored_log_boltzmann_factor_per_outerloop, ns_loaded)


# =============================================================================
# Calculate multimer partition functions
# =============================================================================
# Keep track of the partition functions 
multimers_stored_log_boltzmann_factor = dict()

# Keep track of the partition function including outerloops
multimers_stored_log_boltzmann_factor_with_outerloop = dict()

# Need also to account for each boltzmann factor of each possible outerloop 
multimers_stored_log_boltzmann_factor_per_outerloop = dict()
# multimers_stored_log_boltzmann_factor_per_outerloop[n_vec] is itself a dictionary.
# multimers_stored_log_boltzmann_factor_per_outerloop[n_vec][x] gives
# the partition function for the ensemble of structures comprised of m strands of lengths
# given by n_vec, constrained on them having an outerloop of length x. Includes the partition fxn
# contribution of the outerloop itself as well. When x is np.inf, that refers to the ensemble
# of structures whose outerloop gives no loop entropy contribution


def multimers(n_vec):
    '''
    Input: a tuple (n_1, n_2, ..., n_m)
    Returns: 
        float: partition function for ensemble of structures a group of m strands
            of lengths given by n_vec can form
        float: the partition function including the outerloops. In other words, 
            the partition function of strands of lengths (n_1 + 1, n_2, ..., n_m + 1)
            assuming the first and last nts are bound
        dict: the partition function for each different outerloop length, including
            the cost of forming the outerloop
            
    Calculates the partition function of a group of m strands of lengths given
    by n_vec. Functions by recursively calling itself on strands of lengths
    (n_1, n_2, ..., n_m - 1), and then adding the ensembles of structures for 
    each nt the last nt can bind to.
    
    n = number of stickers (equal to the number of repeats if the repeats are
                            e.g. GCA, but equal to 1 + the number of repeats if
                            the repeats are e.g. CAG).
                            n=0 means 0 stickers; n=1 means 1 sticker, etc.
    '''
    
    m = len(n_vec)
   
    # If we've done this calculation already, return its results
    if n_vec in multimers_stored_log_boltzmann_factor.keys():
        return(multimers_stored_log_boltzmann_factor[n_vec], 
               multimers_stored_log_boltzmann_factor_with_outerloop[n_vec],
               multimers_stored_log_boltzmann_factor_per_outerloop[n_vec])
    
    
    # Treat the base cases: cases that can only form free strands
    base_cases = [(0,), (1,)]
    if not adjacent_stickers_can_bind:
        base_cases += [(2,)]
        
    if n_vec in base_cases:
        n = n_vec[0]
        len_loop = (n + 1) * len_linker + n * len_repeat_bp + 1
        multimers_stored_log_boltzmann_factor[n_vec] = 0.
        multimers_stored_log_boltzmann_factor_with_outerloop[n_vec] = (
                loopEntropyFxn(len_loop) / kB - FE_from_one_bp / (kB * T))
        multimers_stored_log_boltzmann_factor_per_outerloop[n_vec] = {
                len_loop: loopEntropyFxn(len_loop) / kB - FE_from_one_bp / (kB * T)}
        return(multimers_stored_log_boltzmann_factor[n_vec], 
               multimers_stored_log_boltzmann_factor_with_outerloop[n_vec],
               multimers_stored_log_boltzmann_factor_per_outerloop[n_vec])
    
    
    # If there's a zero in n_vec, then it's really not looking at an m-mer
    n_vec_without_0s = tuple([i for i in n_vec if i != 0])
    if n_vec_without_0s != n_vec:
        if not n_vec_without_0s:  # if it's empty
            multimers_stored_log_boltzmann_factor[n_vec] = 0
            multimers_stored_log_boltzmann_factor_with_outerloop[n_vec] = (
                    - FE_from_one_bp / (kB * T))
            multimers_stored_log_boltzmann_factor_per_outerloop[n_vec] = {np.inf: 
                multimers_stored_log_boltzmann_factor_with_outerloop[n_vec]}
            return(multimers_stored_log_boltzmann_factor[n_vec], 
               multimers_stored_log_boltzmann_factor_with_outerloop[n_vec],
               multimers_stored_log_boltzmann_factor_per_outerloop[n_vec])

        multimers_stored_log_boltzmann_factor[n_vec] = multimers(n_vec_without_0s)[0]
        
        multimers_stored_log_boltzmann_factor_with_outerloop[n_vec] = (
                multimers_stored_log_boltzmann_factor[n_vec] - FE_from_one_bp / (kB * T))
        multimers_stored_log_boltzmann_factor_per_outerloop[n_vec] = {np.inf: 
            multimers_stored_log_boltzmann_factor_with_outerloop[n_vec]}

        return(multimers_stored_log_boltzmann_factor[n_vec], 
               multimers_stored_log_boltzmann_factor_with_outerloop[n_vec],
               multimers_stored_log_boltzmann_factor_per_outerloop[n_vec])
        
    
    # Now do the ensemble calculation.
    
    # First consider the ensemble of structures with the last sticker unbound
    n_vec_with_last_nt_unbound = n_vec[:-1] + (n_vec[-1] - 1,)  # it is similar to an (n1, n2, ..., nm-1)-mer
    log_boltzmann_factors = [multimers(n_vec_with_last_nt_unbound)[0]]  # its partition function is the same
    log_boltzmann_factor_per_outerloop = dict()  # but each outerloop is len_repeat longer
    for outerloop in multimers(n_vec_with_last_nt_unbound)[2].keys():
        add_to_outerloop = len_repeat
        # Don't need an if statement here since log_boltzmann_factor_per_outerloop is empty
        log_boltzmann_factor_per_outerloop[outerloop + add_to_outerloop] = (
                multimers(n_vec_with_last_nt_unbound)[2][outerloop] -
                loopEntropyFxn(outerloop) / kB + loopEntropyFxn(outerloop + add_to_outerloop) / kB)
    

    # Now consider ensemble of structures for each nt to which the last nt can bind   
    for strand in range(m):  # which strand it's bound to
        if not adjacent_stickers_can_bind:
            # can bind to any sticker except previous and itself
            num_stickers_it_can_bind = n_vec[strand] - 2 * (strand == m-1)  
        else: 
            # can bind to any sticker except itself
            num_stickers_it_can_bind = n_vec[strand] - (strand == m-1)  
            
        for bp_1 in range(num_stickers_it_can_bind):  
            
            # This base pair splits the RNA into two indep parts
            left_n_vec = n_vec[:strand] + (bp_1,)
            if strand == m - 1:
                inner_n_vec = (n_vec[strand] - bp_1 - 2,)
            else:
                inner_n_vec = (n_vec[strand] - bp_1 - 1,) + n_vec[strand + 1: -1] + (n_vec[-1] - 1,)
            
            # partition function is the sum of the left_n_vec and the inner_n_vec with outerloop
            log_boltzmann_factors.append(  
                    multimers(left_n_vec)[0] + multimers(inner_n_vec)[1])
    
            # In order to calculate the free energy including outerloop, need to know outerloop of left side
            for outerloop in multimers(left_n_vec)[2].keys():
                log_Z_outerloop = multimers(left_n_vec)[2][outerloop]  # includes the cost of the outerloop
                
                 # the outerloop will be extended by an extra linker compared to the outerloop of the left side
                add_to_outerloop = len_linker + 1 
                full_outerloop = outerloop + add_to_outerloop
                
                log_boltzmann_factor_with_outerloop = (
                        log_Z_outerloop +  # the partition function of the left side constrained by the outerloop
                        multimers(inner_n_vec)[1]  # the partition function of the inner part with outerloop (just as before)
                        - loopEntropyFxn(outerloop) / kB +  # the new outerloop is different, so correct for that.
                        loopEntropyFxn(full_outerloop) / kB 
                        )
                
                # Add the ensemble to the partition function constrained on outerloops for the n_vec
                if full_outerloop in log_boltzmann_factor_per_outerloop.keys():
                    log_boltzmann_factor_per_outerloop[full_outerloop] = log_sum_exp(
                            [log_boltzmann_factor_per_outerloop[full_outerloop],
                             log_boltzmann_factor_with_outerloop])
                else:
                    log_boltzmann_factor_per_outerloop[full_outerloop] = (
                            log_boltzmann_factor_with_outerloop)
            
    
    # Add up all the ensembles properly
    multimers_stored_log_boltzmann_factor[n_vec] = log_sum_exp(log_boltzmann_factors)
    multimers_stored_log_boltzmann_factor_with_outerloop[n_vec] = log_sum_exp(
            [i for i in log_boltzmann_factor_per_outerloop.values()])    
    multimers_stored_log_boltzmann_factor_per_outerloop[n_vec] = log_boltzmann_factor_per_outerloop
    
    return(multimers_stored_log_boltzmann_factor[n_vec],  # the log partition function with no outerloop (float)
           multimers_stored_log_boltzmann_factor_with_outerloop[n_vec],  # the log partition function including outerloop (float)
           multimers_stored_log_boltzmann_factor_per_outerloop[n_vec])  # the log partition functions for each outerloop (dict)
            

# =============================================================================
# Compute landscapes
# =============================================================================
max_m = 15  # max size complex to compute
# ns_to_compute = [100] + [60] * 1 + [40] * 1 + [30] * 3 + [21] * 9  # for each m, what should max_n be?
ns_to_compute = [30] * 3 + [20] * 3 + [11] * 9

time_to_compute_multimers = np.zeros((max_m + 1, max(ns_to_compute)))
time_estimate_for_all_multimers = 0

if max_m >= 6:
    time_estimate_for_last_6mer = 1e-4 * ns_to_compute[5]**4.5
    time_estimate_for_all_6mers = np.sum([1e-4 * i**4.5 for i in range(ns_to_compute[5])])
    print('Estimated time to compute final 6-mer = ' + str(time_estimate_for_last_6mer))
    print('Estimated time to compute all 6-mers = ' + str(time_estimate_for_all_6mers))
    time_estimate_for_all_multimers += time_estimate_for_all_6mers
if max_m >= 5:
    time_estimate_for_last_5mer = 6e-5 * ns_to_compute[4]**4.5
    time_estimate_for_all_5mers = np.sum([6e-5 * i**4.5 for i in range(ns_to_compute[4])])
    print('Estimated time to compute final 5-mer = ' + str(time_estimate_for_last_5mer))
    print('Estimated time to compute all 5-mers = ' + str(time_estimate_for_all_5mers))
    time_estimate_for_all_multimers += time_estimate_for_all_5mers
if max_m >= 4:
    time_estimate_for_last_4mer = 4e-5 * ns_to_compute[3]**4.5
    time_estimate_for_all_4mers = np.sum([4e-5 * i**4.5 for i in range(ns_to_compute[3])])
    print('Estimated time to compute final 4-mer = ' + str(time_estimate_for_last_4mer))
    print('Estimated time to compute all 4-mers = ' + str(time_estimate_for_all_4mers))
    time_estimate_for_all_multimers += time_estimate_for_all_4mers
if max_m >= 3:
    time_estimate_for_last_3mer = 1.5e-5 * ns_to_compute[2]**4.5
    time_estimate_for_all_3mers = np.sum([1.5e-5 * i**4.5 for i in range(ns_to_compute[2])])
    print('Estimated time to compute final 3-mer = ' + str(time_estimate_for_last_3mer))
    print('Estimated time to compute all 3-mers = ' + str(time_estimate_for_all_3mers))
    time_estimate_for_all_multimers += time_estimate_for_all_3mers


print('Estimated time to compute all 3+-mers = ' + str(time_estimate_for_all_multimers))


if tryToLoadVariables:
    (multimers_stored_log_boltzmann_factor, multimers_stored_log_boltzmann_factor_with_outerloop, 
     multimers_stored_log_boltzmann_factor_per_outerloop, ns_loaded) = loadFxn(max_m)
else:
    ns_loaded = [0] * max_m

if any([i < j for i, j in zip(ns_loaded, ns_to_compute)]):
    for m in range(1, max_m + 1):
        for n in range(ns_to_compute[m - 1]):
            start = time.time()
            _ = multimers((n,) * m)
            time_to_compute_multimers[m - 1, n] = max(
                    time_to_compute_multimers[m-1, n], time.time() - start)
            if (n+1) % 1 == 0:
                print(str(m) + '-mers: n = ' + str(n) + '; time elapsed = ' + 
                      str(time_to_compute_multimers[m - 1, n]))
                
    if toSave:
        saveFxn(multimers_stored_log_boltzmann_factor, multimers_stored_log_boltzmann_factor_with_outerloop, 
                multimers_stored_log_boltzmann_factor_per_outerloop, ns_to_compute[:max_m])
        print('Finished saving results')

# =============================================================================
# Compute landscapes for single complexes
# =============================================================================
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


"""
In the previous step, we enumerated the *complete* partition function for a group of
strands. Now ask: for each m and each n, what is the partition function
of a group of m strands of length n forming a single complex?

We need for each m-mer to divide the partition function we calculate naively by m.
This accounts for m different ways of depicting each equivalent structure in our model
(i.e. cyclic permutations). There in general would be m! ways of depicting each structure
(the number of ways to permute the strands) but (m-1)! of them result in structures
that look like pseudoknots and are therefore not enumerated. 
After dividing the partition function of each m-mer by m, we need to make sure this
division doesn't make its way into the subtraction part (of subtracting out the 
contributions of e.g. dimers + monomers from the m=3 case) so we need to multiply
back that correction for each subtraction. 

We also use the same trick as in log_sum_exp to perform the computations so that 
we don't get overflow errors

We could do this calculation by hand, as in this commented section, but it's 
much more reasonable to write a general function, which comes next
"""
# complexes = np.zeros((max_m, max(ns_to_compute)))  
# for n in range(1, ns_to_compute[0]):
#     complexes[0][n] = multimers((n,))[0]
    

# if max_m >=2:
#     for n in range(1, ns_to_compute[1]):
#         complexes[1][n] = (log_subtract_exp(
#                 multimers((n, n))[0], 
#                 2 * multimers((n,))[0]) # 2 monomers
#                     - np.log(2))  # correct for overcounting (and account for symmetries)
    
    
# if max_m >= 3:
#     for n in range(2, ns_to_compute[2]):  # need n>=2 to form m-mers where m>=3
#         complexes[2][n] = (log_subtract_exp(
#                 multimers((n, n, n))[0], 
#                 log_sum_exp(
#                         [3 * complexes[0][n],  # 3 monomers
#                          np.log(3) + complexes[0][n] + complexes[1][n] + np.log(2)  # 1 monomer, 1 dimer
#                          # 3 ways to arrange; 1 monomer; 1 dimer; dimer needs to be corrected by factor of 2
#                 ])) - np.log(3))

    
# if max_m >=4:
#     for n in range(2, ns_to_compute[3]):
#         complexes[3][n] = (log_subtract_exp(
#                 multimers((n, n, n, n))[0], 
#                 log_sum_exp(
#                         [4 * complexes[0][n],  # 4 monomers
#                          np.log(4) + complexes[0][n] + complexes[2][n] + np.log(3),  # 1 trimer, 1 monomer
#                          np.log(6) + 2 * complexes[0][n] + complexes[1][n] + np.log(2), # 2 monomers, 1 dimer
#                          np.log(2) + 2 * (complexes[1][n] + np.log(2)) # 2 dimers
#                         # The dimer xyxy can't form because that looks like a pseudoknot! 
#                         # That's why it's only 2 dimers and not 3 (xxyy, xyyx, NOT xyxy)
#                 ])) - np.log(4))

    
# if max_m >= 5:
#     for n in range(2, ns_to_compute[4]):
#         complexes[4][n] = (log_subtract_exp(
#                 multimers((n, n, n, n, n))[0], 
#                 log_sum_exp([
#                         5 * complexes[0][n],   # 5 monomers
#                         np.log(5) + complexes[0][n] + complexes[3][n] + np.log(4),  # 1 monomer, 1 quatromer
#                         np.log(10) + 2 * complexes[0][n] + complexes[2][n] + np.log(3),  # 2 monomers, 1 trimer
#                         np.log(10) + 3 * complexes[0][n] + complexes[1][n] + np.log(2),  # 3 monomers, 1 dimer
#                         np.log(5) + complexes[1][n] + np.log(2) + complexes[2][n] + np.log(3),  # 1 dimer, 1 trimer
#                         # Only 5 (xxyyy, xyyyx, yyyxx, yxxyy, yyxxy). Should be 10 but rest are 'pseudoknots'
#                         np.log(10) + complexes[0][n] + 2 * (complexes[1][n] + np.log(2)), # 2 dimers, 1 monomer
#                         # Only 10 and not 15: (5 for each place the monomer can be in 2 dimer case)
#                 ])) - np.log(5))


# if max_m >= 6:
#     for n in range(2, ns_to_compute[5]):
#         complexes[5][n] = (log_subtract_exp(
#                 multimers((n, n, n, n, n, n))[0],
#                 log_sum_exp([
#                         6 * complexes[0][n],  # 6 monomers
#                         np.log(6) + complexes[0][n] + complexes[4][n] + np.log(5),  # 1 monomer, 1 pentamer
#                         np.log(6) + complexes[1][n] + np.log(2) + complexes[3][n] + np.log(4),  # 1 dimer, 1 quatromer
#                         # Only 6 and not 15: (xxyyyy, yxxyyy, yyxxyy, yyyxxy, yyyyxx, xyyyyx)
#                         np.log(15) + 2 * complexes[0][n] + complexes[3][n] + np.log(4),  # 2 monomers, 1 quatromer
#                         np.log(20) + 3 * complexes[0][n] + complexes[2][n] + np.log(3),  # 3 monomers, 1 trimer
#                         np.log(30) + complexes[0][n] + complexes[1][n] + np.log(2) + complexes[2][n] + np.log(3),  # 1 monomer, 1 dimer, 1 trimer
#                         # Only 30 and not 60 (6*number of ways of making just a dimer and trimer)
#                         np.log(3) + 2 * (complexes[2][n] + np.log(3)),  # 2 trimers
#                         # Only 3 and not 10 (xxxyyy, xxyyyx, xyyyxx)
#                         np.log(5) + 3 * (complexes[1][n] + np.log(2)),  # 3 dimers
#                         # Only 5 and not 15 (xxyyzz, xxyzzy, xyyxzz, xyyzzx, xyzzyx)
#                         np.log(30) + 2 * complexes[0][n] + 2 * (complexes[1][n] + np.log(2)),  # 2 monomers, 2 dimers
#                         # Only 30 and not 45. 15=6*5/2 ways to arrange monomers for each of (xxyy, xyyx, xyxy)
#                         np.log(15) + 4 * complexes[0][n] + 1 * (complexes[1][n] + np.log(2))  # 4 monomers, 1 dimer
#                 ])) - np.log(6))


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
    if m > 15:  # too large -- count all subdivisions msy issues
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



def create_complexes_from_multimers_general(ns_to_compute):
    max_m = len(ns_to_compute)
    complexes = np.zeros((max_m, max(ns_to_compute)))
    
    for n in range(1, ns_to_compute[0]):
        complexes[0][n] = multimers((n,))[0]
        
    for m in range(2, max_m + 1):
        disconnected_possibilities, disconnected_multiplicities = ways_to_connect_m(m)
        for n in range(1 + (m>2), ns_to_compute[m - 1]):
            correction = []
            for possibility, multiplicity in zip(disconnected_possibilities, disconnected_multiplicities):
                correction += [np.log(multiplicity) + sum(
                    [complexes[i - 1][n] + np.log(i) for i in possibility])]
            complexes[m - 1][n] = (log_subtract_exp(
                multimers((n,) * m)[0],  
                log_sum_exp(correction)) - np.log(m))
    return(complexes)
    

complexes = create_complexes_from_multimers_general(ns_to_compute)

# =============================================================================
# Also include the translational entropy cost of binding (\Delta G_assoc)
# =============================================================================
conc = 1e-6 # in units of M
rhoH20 = 55.14  # the units with which we measure concentration, following NUPACK, and Dirks et al. 2007
translational_free_energy_cost_of_binding = 4.09 - kB * T * np.log(rhoH20)
print('Free energy cost of binding = ' + str(translational_free_energy_cost_of_binding))


corrected_log_Z = np.zeros((max_m, max(ns_to_compute)))
for m in range(max_m):
    corrected_log_Z[m, 2:ns_to_compute[m]] = (complexes[m, 2:ns_to_compute[m]] -
                   m * translational_free_energy_cost_of_binding / (kB * T))

# =============================================================================
# Plot results
# =============================================================================
multimer_names = ['monomers', 'dimers', 'trimers', 'tetramers', 'pentamers', 
                  'hexamers', 'septamers', 'octamers', 'nonamers', 'decamers'
                  ] + [str(i) + '-mers' for i in range(11, 21)]
if max_m <= 8:
    multimer_colors = ['b','orange','g','r','purple','brown', 'k', 'cyan', 'magenta', 'yellow']
else:
    import pylab    
    cm = pylab.get_cmap('gist_rainbow')
    multimer_colors = [cm(1.*i/max_m) for i in range(max_m)]


plt.figure()
for m in range(max_m):
    plt.plot(range(2, ns_to_compute[m]), 
             corrected_log_Z[m, 2:ns_to_compute[m]], 
             label=multimer_names[m])
plt.ylabel('log Z')
plt.xlabel('number of repeats')
plt.legend(loc='upper left', fontsize=14)
plt.xlim([2,15])
plt.ylim([-30,100])
plt.show()


plt.figure()
for m in range(1, max_m):
    plt.plot(range(2, ns_to_compute[m]), 
             corrected_log_Z[m, 2:ns_to_compute[m]] - corrected_log_Z[m-1, 2:ns_to_compute[m]], 
             label='m = ' + str(m + 1) + ' - m = ' + str(m))
plt.plot(range(2, ns_to_compute[0]), complexes[0, 2:ns_to_compute[0]], label='m = 1')
if FE_from_one_bp == -10:
    plt.plot(range(2, ns_to_compute[0]), 6.56476 + 3.99764972 * np.arange(2, ns_to_compute[0]), 
             '--k', label='fit')
plt.ylabel('log Z (m) - log Z (m-1)')
plt.xlabel('number of repeats')
plt.title('Single complex GC ensemble')
plt.legend(bbox_to_anchor=(1, 1.05))
plt.show()


plt.figure()
for m in range(3, max_m):
    plt.plot(range(2, ns_to_compute[m]), 
             corrected_log_Z[m, 2:ns_to_compute[m]] - corrected_log_Z[m-1, 2:ns_to_compute[m]], 
             label='m = ' + str(m + 1) + ' - m = ' + str(m))
if FE_from_one_bp == -10:
    plt.plot(range(2, ns_to_compute[0]), 6.56476 + 3.99764972 * np.arange(2, ns_to_compute[0]), 
             '--k', label='fit')
plt.ylabel('log Z (m) - log Z (m-1)')
plt.xlabel('number of repeats')
plt.title('Single complex GC ensemble')
plt.legend(bbox_to_anchor=(1, 1.05))
plt.show()


plt.figure()
for m in range(max_m):
    plt.plot(range(2, min(ns_to_compute)), 
             (1 / (m+1)) * corrected_log_Z[m, 2:min(ns_to_compute)], 
             linestyle='--',
             label=multimer_names[m])
plt.ylabel('log Z / m')
plt.xlabel('number of repeats')
plt.legend(bbox_to_anchor=(1, 1.05))
plt.show()




plt.figure()
for n in range(2, min(ns_to_compute)):
    plt.plot(range(1, max_m + 1), 
             corrected_log_Z[:, n], 
             label='n = ' + str(n))
plt.ylabel('log Z')
plt.xlabel('number of strands in structure')
plt.legend(bbox_to_anchor=(1.05,1))
plt.show()



plt.figure()
for m in range(1, max_m):
    plt.plot(range(2, ns_to_compute[m]), 
             corrected_log_Z[m, 2:ns_to_compute[m]] / (m + 1) - corrected_log_Z[0, 2:ns_to_compute[m]], 
             label=multimer_names[m])
plt.ylabel('log Z (m) / m - log Z (1)')
plt.xlabel('number of repeats')
plt.legend(loc='upper right', fontsize=14)
plt.show()



plt.figure()
for n in [3, 7, 11, 15, 19, 4, 8, 12, 16, 20]:  # range(2, min(ns_to_compute)):
    plt.plot(range(2, max_m + 1), 
             corrected_log_Z[1:, n] / np.arange(2, max_m + 1) - corrected_log_Z[0, n], 
             label='n = ' + str(n))
plt.ylabel('log Z (m) / m - log Z (1)')
plt.xlabel('number of strands in structure')
plt.legend(bbox_to_anchor=(1.05,1))
plt.show()