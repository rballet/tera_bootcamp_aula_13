# coding: utf-8

import numpy as np


def has_all_outcomes(list1, list2):
    return all(map(lambda v: v in list2, list1))
    

population_set = [1,2,3,4,5,6]
mean_trials = []
for t in range(1000):
    n_trials_limit = []
    for j in range(100):
        limit_found = False
        outcome = []
        i = 0
        while not limit_found:
            i += 1
            outcome.append(np.random.choice(population_set))
            if has_all_outcomes(population_set, outcome):
                n_trials_limit.append(i)
                limit_found = True
    mean_trials.append(np.mean(n_trials_limit))
    
    
def throw_dice():
    return np.random.choice(population_set)
    
    
def throw_until_all_faces():
    outcome = []
    i = 0
    limit_found = False
    while not limit_found:
        i += 1
        outcome.append(throw_dice())
        if has_all_outcomes(population_set, outcome):
            n_trials_limit.append(i)
            limit_found = True
    return i, outcome
    

def get_n_trials_distribution(n=100):
    n_trials_limit = []
    for j in range(n):
        i, _ = throw_until_all_faces()
        n_trials_limit.append(i)
    return n_trials_limit
    
    
def get_mean_trials_distribution(n=1000):
    mean_trials = []
    for t in range(n):
        n_trials = get_n_trials_distribution()
        mean_trials.append(np.mean(n_trials))
    return mean_trials