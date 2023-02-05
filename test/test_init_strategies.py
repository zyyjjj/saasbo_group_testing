import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append(['..'])
sys.path.append("/home/yz685/saasbo_group_testing")

import numpy as np
import torch
from botorch.test_functions.synthetic import Hartmann

from saasbo_group_testing.init_strategies import (perturb_input_dims,
                                                  sequential_bifurcation,
                                                  split_range)
from saasbo_group_testing.test_problems.embedded_test_problem import \
    EmbeddedTestProblem

print("===== Testing split_range() =====")
print(split_range([1,2,3,4,5], 2))
print(split_range([1,2,3,4], 2))
print(split_range([1,2,3,4,5,6], 3))
print(split_range([1,2,3,4,5,6,7,8], 3))


print("===== Testing perturb_input_dims() =====")
print(perturb_input_dims(torch.tensor([0.5, 0.5, 0.5]), [0, 1], "random"))
print(perturb_input_dims(torch.tensor([0.5, 0.5, 0.5]), [0, 1], "ub"))
print(perturb_input_dims(torch.tensor([0.5, 0.5, 0.5]), [0, 1], "lb"))


print("===== Testing sequential_bifurcation() =====")
problem = Hartmann()
embedded_problem = EmbeddedTestProblem(input_dim = 50, base_problem=problem)
print('embedded indices: ', embedded_problem.emb_indices)
print(embedded_problem(torch.rand(50)))
X, Y, important_dims = sequential_bifurcation(
    problem=embedded_problem, 
    perturb_option="ub", 
    n_folds=2, 
    seed=0, 
    verbose=True)
