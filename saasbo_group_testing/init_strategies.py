from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

# Helper functions

def split_range(s: List, n_folds: int = 2) -> List[List]:
    r"""Split a list of numbers into `n_folds` equally sized chunks 
    (up to rounding). 
    Return them as a list, storing the chunks from right to left.
    Args:
        s: list of numbers
        n_folds: number of chunks we want to split list s into
    Returns:
        split_indices: nested list with `n_folds` sublists of s 
            (mutually exclusive, collectively exhaustive, sizes differ by <= 1)
    """

    if len(s) <= n_folds:
        # just return elements of s one by one
        split_indices = [[s_] for s_ in s]

    else:
        # have a scheme for splitting the list 
        n_incomplete_chunks = n_folds
        base_chunk_size = len(s) // n_folds
        n_incomplete_larger_chunks = len(s) % n_folds

        split_indices = []
        start_idx = 0
        
        while n_incomplete_chunks > 0:
            if n_incomplete_larger_chunks > 0:
                end_idx = start_idx + base_chunk_size + 1
                n_incomplete_larger_chunks -= 1
            else:
                end_idx = start_idx + base_chunk_size
            
            split_indices.insert(0, s[start_idx : end_idx])

            start_idx = end_idx
            n_incomplete_chunks -= 1
        
        # when the loop terminates, we should exactly exhaust the list
        assert end_idx == len(s), "end index does not match length of list s"

    return split_indices

def perturb_input_dims(
    status_quo_input: Tensor, 
    dims_to_perturb: list, 
    perturb_option: str,
    seed: Optional[int] = None
) -> Tensor:
    r"""Perturb specified dimensions of a given status quo tensor, while
    leaving the other dimensions intact.
    Args:
        status_quo_input: baseline input value to perturb from
        dims_to_perturb: list of dimensions to perturb -- # TODO: can we do this in batch
        perturb_option: one of {'random', 'ub', 'lb'}
    """

    if seed is not None:
        torch.manual_seed(seed)

    if perturb_option == "random":
        perturb_vals = torch.rand(len(dims_to_perturb))
    elif perturb_option == "ub":
        perturb_vals = torch.ones(len(dims_to_perturb))
    elif perturb_option == "lb":
        perturb_vals = torch.zeros(len(dims_to_perturb))
    # TODO: allow input domains that aren't necessarily [0,1]
    
    perturbed_input = status_quo_input.detach().clone()
    perturbed_input[dims_to_perturb] = perturb_vals

    return perturbed_input


# Initialization strategies

def sequential_bifurcation(
    problem: torch.nn.Module, 
    perturb_option: str, 
    n_folds: int = 2,
    seed: Optional[int] = None,
    verbose: bool = True
) -> Tuple[Tensor, Tensor, List]:
    r"""Implement the initialization strategy `sequential bifurcation` to 
    identify important input dimensions adaptively.
    Let d denote the input dimension of problem.

    Concretely, the algorithm proceeds as follows:
    - first evaluate the problem at the center of the input domain x0; 
        call the observed outcome value y0; store this datapoint
    - have an empty stack for sets of dimensions to perturb,
    - initialize the stack with `n_folds` equally sized sublists of range(d),
        ordered from right to left
    - while the stack is not empty:
        - pop the last element s from the stack
        - evaluate the problem at an input point where the input dims in s are 
            perturbed (either random or set to upper bound) and other input dims
            stay at the center of the domain; store this datapoint
        - if f(s) != y0: 
            - if s has >1 element: append `n_folds` equally sized sublists of 
                s to the stack, ordered from right to left
            - elif s has 1 element: record the element in s as important
    - return the evaluated input points, observed values, and important dims

    NOTE: 
    When `n_folds=2`, this is equivalent to preorder traversal (root-left-right) 
    of a binary tree. The nodes of the tree are the sets of indices to perturb; 
    a node is a leaf node if it does not contain any important dims; a node 
    further branches into two children if it contains at least one important dim.
    Since we append the right child first to the stack, we pop out and evaluate
    the left child before the right child, resulting in root-left-right order.

    Args:
        problem: high-dimensional test problem
        perturb_option: one of {'random', 'ub', 'lb'} to perturb the input dims.
            If 'random', set to numbers sampled uniform at random from the 
                respective domains of the dims;
            if 'ub', set to upper bounds of the respective domains of the dims;
            if 'lb', set to lower bounds of the respective domains of the dims. 
        n_folds: number of folds to split each set of indices into. Default 2. 
    Returns:
        X: tensor of evaluated inputs
        Y: tensor of observations at evaluated inputs
        important_dims: list of identified important dims
    """

    stack, X, Y, important_dims = [], [], [], []
    
    x0 = torch.tensor(problem._bounds).mean(dim=1) # shape (`input_dim`,)
    y0 = problem(x0) # TODO: make sure we start with noiseless evals
    X.append(x0)
    Y.append(y0)

    print("observation of baseline input: ", y0)

    stack += split_range(list(range(problem.dim)))

    while stack:
        s = stack.pop()
        
        x = perturb_input_dims(
            status_quo_input = x0, 
            dims_to_perturb = s, 
            perturb_option = perturb_option, 
            seed=seed)
        y = problem(x)

        if verbose: 
            print("popped set of indices: ", s)
            print("evaluation of perturbed x: ", y)
        
        X.append(x)
        Y.append(y)
        
        if y != y0: 
        # TODO: alternatively, if error norm is less than some tolerance
            if verbose: 
                print("There exists >=1 important dim in current set of indices")
            if len(s) == 1:
                print(f"Identified {s[0]} as important dim!")
                important_dims.append(s[0])
            else:
                s_split = split_range(s, n_folds)
                stack += s_split
        else:
            if verbose:
                print("No important dim in current set of indices")
    
    # turn list of tensors into tensor
    X = torch.stack(X)
    Y = torch.stack(Y)

    return X, Y, important_dims


def random_subset(
    problem: torch.nn.Module, 
    n_samples: int, 
    perturb_option: str, 
    n_folds: int = 2
):
    r"""Implement the initialization strategy `random subset` to 
    identify important input dimensions.
    The algorithm proceeds as follows:
    - first evaluate the problem at the center of the input domain x0; 
        call the observed outcome value y0; store this datapoint
    - generate `n_samples` perturbed input points, where each input point is 
        generated by perturbing a fraction 1/n_folds of all dimensions from x0
    - evaluate the problem at the perturbed input points and store the outcomes
    - fit a regularized linear regression model with L1 penalty on the data
    - return the set of input dimensions with nonzero regression coefficient
    
    Args:
        problem:
        n_samples:
        perturb_option:
        n_folds:
    Returns:
        X:
        Y:
        important_dims:
    """
    
    # TODO: choosing hyperparameter lambda in lasso

    pass