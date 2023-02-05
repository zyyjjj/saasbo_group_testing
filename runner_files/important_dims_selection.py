import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append(['..'])
sys.path.append("/home/yz685/saasbo_group_testing")

import numpy as np
import torch
from botorch.test_functions import Hartmann

from saasbo_group_testing.experiment import Experiment
from saasbo_group_testing.init_strategies import (perturb_input_dims,
                                                  sequential_bifurcation,
                                                  split_range)
from saasbo_group_testing.test_problems.embedded_test_problem import \
    EmbeddedTestProblem

if __name__ == "__main__":

    base_problem = Hartmann()
    embedded_problem = EmbeddedTestProblem(input_dim = 50, base_problem=base_problem)

    print("embedded_problem.dim: ", embedded_problem.dim)
    print("base problem dim: ", base_problem.dim)

    print(embedded_problem(torch.rand(50)))

    X, Y, important_dims = sequential_bifurcation(
        problem = embedded_problem, perturb_option = 'ub', n_folds = 2)

    print("inferred important dims: ", important_dims)
    print("true important dims: ", embedded_problem.emb_indices)
    print(
        "correctly inferred ", 
        len(set(embedded_problem.emb_indices).intersection(important_dims)), 
        f" important dims out of {base_problem.dim}"
    )
    print("number of samples gathered: ", len(X))