import os
import sys

import numpy as np
import torch
from botorch.test_functions import Hartmann
from botorch.test_functions.base import BaseTestProblem


class EmbeddedTestProblem(BaseTestProblem):
    
    r"""
    Class for test problems embedded into higher dimensional input space.
    """

    def __init__(
        self,
        input_dim: int,
        base_problem: torch.nn.Module,
        noise_std: float = 0,
    ):
        r"""
        Args:
            input_dim: the dimensionality of input space to embed the problem into
            base_problem: a test problem without artificially raising input dim
            noise_std: magnitude of independent noise to add to input dims
        """


        # sample embedding indices for the base input dims from range(input_dim)
        emb_indices = np.random.choice(
            input_dim, size=base_problem.dim, replace=False)

        # create bounds tensor, imputing the extra dims with [0,1]
        self._bounds = torch.Tensor([[0,1]] * input_dim)
        for i in range(base_problem.dim):
            self._bounds[emb_indices[i], :] = torch.Tensor(base_problem._bounds[i])

        super().__init__(
            noise_std = noise_std, 
            negate = base_problem.negate
        ) 

        self.dim = input_dim


        self.base_problem = base_problem
        self.emb_indices = emb_indices
        

        # TODO: how to deal with seed here?

    
    def evaluate_true(self, X: torch.Tensor):
        
        # take out the dims that matter
        return self.base_problem.evaluate_true(X[..., self.emb_indices])


if __name__ == "__main__":

    problem = Hartmann()
    embedded_problem = EmbeddedTestProblem(input_dim = 50, base_problem=problem)

    print('embedded indices: ', embedded_problem.emb_indices)

    print(embedded_problem(torch.rand(50)))