import os
import sys
import time

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append(['..'])

import numpy as np
import torch
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples

from saasbo_group_testing.init_strategies import (random_subset,
                                                  sequential_bifurcation)


class Experiment:

    r"""
    Runs one instance of a BO experiment.
    """

    attr_list = {
        "n_bo_iters": 20,
        "verbose": False,
        "maxiter": 1000,
        "dtype": torch.double,
        "raw_samples": 128,
        "num_restarts": 20,
        "batch_limit": 4,
        "n_initial_samples": 32
    }


    def __init__(
        self,
        problem: torch.nn.Module,
        init_strategy: str,
        selection_strategy: str,
        seed: int,
        output_path: str,
        **kwargs
    ):
        r""" Initialize experiment settings:
        Args:
            problem: test problem with high-dimensional input and scalar output
            init_strategy: string specifying the strategy for 
                generating initial samples for fitting the GP before BO;
                if 'sobol': use Sobol sampling;
                if 'seq_bif': use sequential bifurcation;
                if 'rand_subset_lasso': perturb random subsets of the data and train 
                    a L1-regularized model
            selection_strategy: string specifying the way of passing information 
                about which dimensions are important to the BO step;
                if 'no_select': don't do anything;
                if 'reduce_model': do BO on subset of input dimensions;
                if 'modify_saas_prior': make the lengthscale priors of the 
                    important dims concentrated at smaller values in saas prior
            seed: random seed
            output_path: path to save the output
        """

        for key in self.attr_list.keys():
            setattr(self, key, self.attr_list[key])
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        
    def generate_initial_data(self, n):
        r"""
        Generate (input, outcome) data for initializing the GP before BO
        TODO: this goes hand in hand with identifying important dimensions
        So the goals are: get self.X, self.Y and self.important_dims
        """
        if self.init_strategy == "sobol":
            self.X = draw_sobol_samples(bounds = self.problem.bounds, n=1, q=n).squeeze(0)
            self.X.to(self.dtype)
            self.Y = self.problem(self.X)
        elif self.init_strategy == "seq_bif":
            # TODO: implement sequential bifurcation 
            pass
        elif self.init_strategy == "rand_subset_lasso":
            # TODO: implement this
            pass
    
    
    def fit_gp(self):
        r"""
        Fit GP based on all data so far
        
        """
        # TODO: have proper error handling 

        pass

    
    def pass_importance_information(self):
        r"""
        (Not sure if this is the best way,
        but must remember to implement the selection strategy)
        """

        if self.selection_strategy == "no_select": 
            pass
        elif self.selection_strategy == "reduce_model": 
            pass
            # 
        elif self.selection_strategy == "modify_saas_prior":
            pass


        pass


    def one_bo_iteration(self):
        r"""
        Run one iteration of BO 
        """
        # TODO: or maybe run multiple BO iters in this function
        # TODO: time it
        # TODO: save the data after each iteration
        
        
        start_time = time.time()


    def full_pipeline(self):
        # TODO: overall structure is as follows, to flesh out

        self.generate_initial_data(self.n_initial_samples)

        # call fit_gp() and learn important dimensions
        
        # call pass_importance_information() 
        
        for _ in range(self.n_bo_iters):
            self.one_bo_iteration()