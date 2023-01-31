import torch
import numpy as np

from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples

class Experiment:

    r"""
    Runs one instance of an experiment.
    """

    attr_list = {
        "num_restarts": 20,
        "verbose": False,
        "maxiter": 1000,
        "dtype": torch.double,
        "raw_samples": 128,
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
                if 'modify_prior': make the lengthscale priors of the important dims
                    concentrated at smaller values
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

        pass

    
    def pass_importance_information(self):
        r"""
        (Not sure if this is the best way,
        but must remember to implement the selection strategy)
        """

        pass


    def one_bo_iteration(self):
        r"""
        Run one iteration of BO 
        """
        # TODO: time it
        # TODO: save the data after each iteration
        pass