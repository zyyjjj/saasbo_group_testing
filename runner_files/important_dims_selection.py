import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append(['..'])
sys.path.append("/home/yz685/saasbo_group_testing")

from typing import List

import numpy as np
import torch
from botorch.test_functions import Hartmann

from saasbo_group_testing.src.init_strategies import (perturb_input_dims,
                                                      sequential_bifurcation,
                                                      split_range)
from saasbo_group_testing.src.saasgp import saasgp
from saasbo_group_testing.test_problems.embedded_test_problem import \
    EmbeddedTestProblem


def select_important_dims_seq_bif(
    emb_problem: torch.nn.Module,
    perturb_options: List[str],
    seed: int,
    n_folds: int = 2,
    verbose: bool = False
): 

    res = []

    for perturb_option in perturb_options:

        X, Y, inferred_important_dims = sequential_bifurcation(
            problem=emb_problem, 
            perturb_option=perturb_option, 
            n_folds=n_folds, 
            seed=seed, 
            verbose=verbose)
    
        num_correct = len(set(emb_problem.emb_indices).intersection(inferred_important_dims))

        res.append({
            "perturb_option": perturb_option, 
            "num_correct": num_correct, 
            "num_samples": len(X)
        })

    return res

def select_important_dims_saasgp(
    emb_problem: torch.nn.Module,
    n_samples: int,
    inference_method: str,
    verbose: bool = False
):

    true_important_dims = emb_problem.emb_indices

    saasgp_results = saasgp(
        problem=emb_problem,
        n_samples=n_samples, 
        inference_method=inference_method,
        verbose=verbose,
    )

    res = []

    for datasize, intermediate_results in saasgp_results.items():
        dims_ordered = intermediate_results["dims_ordered"]
        inferred_important_dims = dims_ordered[:len(true_important_dims)].numpy()
        num_correct = len(set(emb_problem.emb_indices).intersection(inferred_important_dims))

        print(f"Fitting saasgp with {datasize} samples, true important dims are ", 
            emb_problem.emb_indices, ", inferred important dims are ", 
            inferred_important_dims, f", {num_correct} dims correctly identified")

        res.append({"num_correct": num_correct, "num_samples": datasize})
    
    return res


def wrapper(
    base_problem_name: str, 
    input_dim: int,
    perturb_options: List[str],
    saasgp_n_samples: int,
    saasgp_inference_method: str = "mle",
    seq_bif_n_folds: int = 2,
    n_trials: int = 10,
    verbose: bool = False
):

    # TODO: allow more problems

    save_name = f"{base_problem_name}_{input_dim}/"
    output_path = '/home/yz685/saasbo_group_testing/experiments/' + save_name
    if not os.path.exists(output_path):
            os.makedirs(output_path)

    if base_problem_name == 'hartmann':
        base_problem = Hartmann()

    for trial_idx in range(n_trials):
        
        print(f"Running trial {trial_idx}")

        emb_problem = EmbeddedTestProblem(
            input_dim=input_dim, base_problem=base_problem, seed=trial_idx)

        res = {"trial_idx": trial_idx, "emb_indices": emb_problem.emb_indices}

        seq_bif_results = select_important_dims_seq_bif(
            emb_problem=emb_problem,
            perturb_options=perturb_options,
            seed=trial_idx,
            n_folds=seq_bif_n_folds,
            verbose=verbose
        )
        res["seq_bif"] = seq_bif_results # list of dicts

        saasgp_results = select_important_dims_saasgp(
            emb_problem=emb_problem,
            n_samples=saasgp_n_samples,
            inference_method=saasgp_inference_method,
            verbose=verbose
        )
        res["saasgp"] = saasgp_results # list of dicts

        torch.save(res, f"{output_path}dim_selection_trial={trial_idx}.th")

if __name__ == "__main__":

    N_TRIALS = 100

    wrapper(
        base_problem_name="hartmann",
        input_dim=50,
        perturb_options=["ub", "lb", "random"],
        saasgp_n_samples=100,
        saasgp_inference_method="mle",
        seq_bif_n_folds=2,
        n_trials=N_TRIALS,
        verbose=False
    )
