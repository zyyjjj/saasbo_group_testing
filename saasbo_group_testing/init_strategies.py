from typing import List

import numpy as np
import torch

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
    status_quo_input: torch.Tensor, 
    dims_to_perturb: list, 
    perturb_option: str
):
    r"""Perturb specified dimensions of a given status quo tensor, while
    leaving the other dimensions intact.
    Args:
        status_quo_input: baseline input value to perturb from
        dims_to_perturb: list of dimensions to perturb -- # TODO: can we do this in batch
        perturb_option: one of {'random', 'ub', 'lb'}
    """

    if perturb_option == "random":
        perturb_vals = torch.rand(len(dims_to_perturb))
    elif perturb_option == "ub":
        perturb_vals = torch.ones(len(dims_to_perturb))
    elif perturb_option == "lb":
        perturb_vals = torch.zeros(len(dims_to_perturb))
    
    perturbed_input = status_quo_input.detach().clone()
    perturbed_input[dims_to_perturb] = perturb_vals

    return perturbed_input


# Initialization strategies

def sequential_bifurcation(
    problem: torch.nn.Module, 
    perturb_option: str, 
    n_folds: int = 2
):
    r"""Implement the initialization strategy `sequential bifurcation` to 
    identify important input dimensions adaptively.
    Let d denote the input dimension of problem.
    Concretely, the algorithm proceeds as follows:
    - first evaluate the problem at the center of the input domain x0; 
        call the observed outcome value y0; store this datapoint
    - have an empty stack for sets of dimensions to perturb,
    - initialize the stack to contain range(d//2) and range(d//2, d)
    - while the stack is not empty:
        - pop the last element s from the stack
        - evaluate the problem at an input point where the input dims in s are 
            perturbed (either random or set to upper bound) and other input dims
            stay at the center of the domain; store this datapoint
        - if f(s) != y0: 
            - if s has >1 element: append two new elements to the stack, which 
                are the first and second half of the entries in s, respectively
            - elif s has 1 element: record the element in s as important
    (NOTE: to make this preorder traversal, append the right half first)
    - return the evaluated input points, observed values, and important dims

    Args:
        problem:
        perturb_option:
        n_folds:
    Returns:
        X:
        Y:
        important_dims:
    """

    stack, X, Y, important_dims = [], [], [], []

    x0 = np.array(problem._bounds.mean(dim = 1)) # shape (`input_dim`,)
    y0 = problem(x0) # TODO: make sure we start with noiseless evals
    X.append(x0)
    Y.append(y0)

    stack.append(split_range(problem.dim//2, problem.dim))
    stack.append(split_range(problem.dim//2))

    while stack:
        s = stack.pop()
        x = perturb_input_dims(x0, s, perturb_option)
        y = problem(x)
        X.append(x)
        Y.append(y)
        
        if y != y0: 
        # TODO: alternatively, if error norm is less than some tolerance
            if len(s) == 1:
                important_dims.append(s[0])
            else:
                s_split = split_range(s, n_folds)
                stack += s_split
    
    # turn list of tensors into tensor
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    
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



if __name__ == "__main__":

    # test split_range()
    # print(split_range([1,2,3,4,5], 2))
    # print(split_range([1,2,3,4], 2))
    # print(split_range([1,2,3,4,5,6], 3))
    # print(split_range([1,2,3,4,5,6,7,8], 3))


    # test perturb_input_dims()
    print(perturb_input_dims(torch.tensor([0.5, 0.5, 0.5]), [0, 1], "random"))
    print(perturb_input_dims(torch.tensor([0.5, 0.5, 0.5]), [0, 1], "ub"))
    print(perturb_input_dims(torch.tensor([0.5, 0.5, 0.5]), [0, 1], "lb"))
