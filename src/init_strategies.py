import numpy as np
import torch


def sequential_bifurcation(problem, n_folds = 2):
    r"""Adopts the initialization strategy `sequential bifurcation` to 
    identify important input dimensions adaptively.
    Let d denote the input dimension of problem.
    Concretely, the algorithm proceeds as follows:
    - first evaluate the problem at the center of the input domain; 
        call the resulting outcome value y0; store this datapoint
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
    """


    pass


def random_subset(problem, n_folds = 2):
    r"""
    
    """
    pass