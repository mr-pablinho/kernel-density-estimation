# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 11:11:09 2021
Testing PCE with arbitray PDF
@author: PMR
"""

import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt

# %% Solver function


def solver(x, y):
    sol = (x**2) + np.sin(y)
    return sol


np.random.seed(123)

section_a = cp.Normal(10, 1.0).sample(100)
section_b = cp.Normal(24, 0.5).sample(100)

points = np.concatenate((section_a, section_b))
np.random.shuffle(points)

coord = np.linspace(0, len(points), len(points))

distribution = cp.GaussianKDE(points, estimator_rule="scott")
cdfs = distribution.cdf(10)


# %% MC

samples = distribution.sample(1000)

evaluations_mc = []
for i in samples:
    y = 10
    x = i
    res = solver(x, y)
    evaluations_mc.append(res)
    
E_mc = np.mean(evaluations_mc)
S_mc = np.std(evaluations_mc)
    

# %% PCE

polyOrder = 10
quadOrder = 10
polynomial_expansion = cp.expansion.stieltjes(polyOrder, distribution, normed=True)
nodes, weights = cp.generate_quadrature(order=quadOrder, dist=distribution, rule="gaussian")

evaluations_pce = []
for i in nodes:
    y = 10
    x = i
    res = solver(x, y)
    evaluations_pce.append(res)
    
    
foo_approx = cp.fit_quadrature(polynomial_expansion, nodes, weights, evaluations_pce[0])

E_pce = cp.E(foo_approx, distribution)
S_pce = cp.Std(foo_approx, distribution)


# %% Figures

# fig 1
plt.figure('1')
plt.hist(points, bins=20)
plt.xlim(0, 40)

# fig 2
plt.figure('2')
plt.hist(samples, bins=20)
plt.xlim(0, 40)



