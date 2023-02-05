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
from saasbo_group_testing.src.init_strategies import (perturb_input_dims,
                                                      sequential_bifurcation,
                                                      split_range)
from saasbo_group_testing.test_problems.embedded_test_problem import \
    EmbeddedTestProblem


def select_important_dims_saasgp(

):

    # either fully Bayesian inference using NUTS 
    # or MLE (add saas prior, then fit gpytorch scipy)

    # should call code under src/saasgp.py


    pass



def select_important_dims_seq_bif(
    base_problem: torch.nn.Module, 
    emb_problem: torch.nn.Module,
    perturb_option: str,
    seed: int,
    n_folds: int = 2,
    verbose: bool = False
): 

    X, Y, important_dims = sequential_bifurcation(
        problem=emb_problem, 
        perturb_option=perturb_option, 
        n_folds=n_folds, 
        seed=seed, 
        verbose=verbose)
    
    print("inferred important dims: ", important_dims)
    print("true important dims: ", emb_problem.emb_indices)
    num_correct = len(set(emb_problem.emb_indices).intersection(important_dims))
    print(
        f"correctly inferred {num_correct} important dims out of {base_problem.dim}"
    )
    print("number of samples gathered: ", len(X))

    return num_correct, len(X), emb_problem.emb_indices


def wrapper(
    base_problem_name: str, 
    input_dim: int,
    perturb_option: str,
    n_folds: int = 2,
    n_trials: int = 10,
    verbose: bool = False
):

    # TODO: allow more problems
    # TODO: allow encoding the selection strategy

    save_name = f"SB_{base_problem_name}_{input_dim}_{perturb_option}/"
    output_path = '/home/yz685/saasbo_group_testing/experiments/' + save_name
    if not os.path.exists(output_path):
            os.makedirs(output_path)

    if base_problem_name == 'hartmann':
        base_problem = Hartmann()

    for trial_idx in range(n_trials):
        res = {}
        print(f"Running trial {trial_idx}")

        emb_problem = EmbeddedTestProblem(
            input_dim=input_dim, base_problem=base_problem, seed=trial_idx)
        
        num_correct, num_samples, emb_indices = select_important_dims_seq_bif(
            base_problem=base_problem,
            emb_problem=emb_problem,
            input_dim=input_dim,
            perturb_option=perturb_option,
            seed=trial_idx,
            n_folds=n_folds,
            verbose=verbose
        )

        # TODO: run emb_problem on saasgp baseline
        # Vary the number of initial samples, look at the 6 dims with smallest LS
        # then expand the res dictionary

        res = {
            "trial_idx": trial_idx,
            "num_correct": num_correct,
            "num_samples": num_samples,
            "emb_indices": emb_indices
        }

        torch.save(res, f"{output_path}dim_selection_trial={trial_idx}.th")

if __name__ == "__main__":

    N_TRIALS = 10
    INPUT_DIM = 50
    PROBLEM_NAME = "hartmann"

    for perturb_option in ["ub", "lb", "random"]:

        wrapper(
            base_problem_name=PROBLEM_NAME,
            input_dim=INPUT_DIM,
            perturb_option=perturb_option,
            n_folds=2,
            n_trials=N_TRIALS
        )
