import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append(['..'])
sys.path.append("/home/yz685/saasbo_group_testing")

from typing import List, Optional

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.fit import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
from botorch.models import FixedNoiseGP
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.optim.fit import fit_gpytorch_scipy
from botorch.test_functions import Branin, Hartmann
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import HalfCauchyPrior
from torch import Tensor

from saasbo_group_testing.src.init_strategies import oracle
from saasbo_group_testing.test_problems.embedded_test_problem import \
    EmbeddedTestProblem


# code credit to David
class SaasPriorHelper:
    """Helper class for specifying parameter and setting closures."""

    def __init__(self, tau: Optional[float] = None):
        self._tau = tau

    def tau(self, m):
        return (
            self._tau
            if self._tau is not None
            else m.raw_tau_constraint.transform(m.raw_tau)
        )

    def inv_lengthscale_prior_param_or_closure(self, m):
        return self.tau(m) / (m.lengthscale ** 2)

    def inv_lengthscale_prior_setting_closure(self, m, value):
        lb = m.raw_lengthscale_constraint.lower_bound
        ub = m.raw_lengthscale_constraint.upper_bound
        m._set_lengthscale((self.tau(m) / value).sqrt().clamp(lb, ub))

    def tau_prior_param_or_closure(self, m):
        return m.raw_tau_constraint.transform(m.raw_tau)

    def tau_prior_setting_closure(self, m, value):
        lb = m.raw_tau_constraint.lower_bound
        ub = m.raw_tau_constraint.upper_bound
        m.raw_tau.data.fill_(
            m.raw_tau_constraint.inverse_transform(value.clamp(lb, ub)).item()
        )

# code credit to David
def add_saas_prior(
    base_kernel: Kernel, tau: Optional[float] = 0.1, **tkwargs
) -> Kernel:
    if not base_kernel.has_lengthscale:
        raise UnsupportedError("base_kernel must have lengthscale(s)")
    if hasattr(base_kernel, "lengthscale_prior"):
        raise UnsupportedError("base_kernel must not specify a lengthscale prior")
    prior_helper = SaasPriorHelper(tau=tau)
    base_kernel.register_prior(
        name="inv_lengthscale_prior",
        prior=HalfCauchyPrior(torch.tensor(1.0, **tkwargs)),
        param_or_closure=prior_helper.inv_lengthscale_prior_param_or_closure,
        setting_closure=prior_helper.inv_lengthscale_prior_setting_closure,
    )
    return base_kernel




def saasgp(
    problem: torch.nn.Module,
    n_samples: int, # TODO: the goal is to log intermediate LS inferences too
    inference_method: str,
    checkpoints: Optional[List] = None,
    verbose: bool = False,
    init_X: Optional[Tensor] = None,
    **kwargs
):
    r"""
    Fit a SAASGP using maximum likelihood estimation.
    Log the inferred important dims as more observations are made.

    Args:
        problem: test problem
        n_init_samples: number of samples to take 
        inference_method: one of {'mle', 'nuts'}
        checkpoints: list of fractions at which we want to pause and examine 
            inferred lengthscales
        verbose: if True, print more information
        init_X: if supplied, use it as the training input
    Returns:
        train_X:
        train_Y:
        important_dims:

    """

    if init_X is not None:
        train_X = init_X
    else:
        train_X = draw_sobol_samples(
            bounds=problem.bounds,
            n=1,
            q=n_samples
        ).squeeze(0)

    train_Y = problem(train_X)
    if len(train_Y.shape) < len(train_X.shape):
        train_Y = train_Y.unsqueeze(1)
        # train_Y = (train_Y - train_Y.mean()) / train_Y.std() # TODO: ?

    print("train_X, train_Y shape: ", train_X.shape, train_Y.shape)

    if checkpoints is None:
        checkpoints = [0.25, 0.5, 0.75, 1]

    if inference_method == "mle":
        base_kernel = MaternKernel(ard_num_dims=problem.dim)
        add_saas_prior(base_kernel, tau=0.1)
        covar_module = ScaleKernel(base_kernel)

    res = {}

    # progressively increase training data size
    for checkpoint in checkpoints:
        datasize = int(n_samples * checkpoint)
        print(f"fitting on first {datasize} points")

        if inference_method == "mle":

            gp = FixedNoiseGP(
                    train_X=train_X[:datasize, :],
                    train_Y=train_Y[:datasize, :],
                    train_Yvar=1e-8 * torch.ones_like(train_Y[:datasize, :]),
                    covar_module=covar_module,
                )
            mll = ExactMarginalLogLikelihood(model=gp, likelihood=gp.likelihood)
            base_kernel._set_lengthscale(1e3)  # Initialize to 1e3, i.e., all being unimportant
            # fit_gpytorch_scipy(
            #     mll, bounds={"model.covar_module.base_kernel.raw_lengthscale": [0.1, 1e3]}
            # )
            fit_gpytorch_mll(mll)

            lengthscales = gp.covar_module.base_kernel.lengthscale

        elif inference_method == "nuts":
            gp = SaasFullyBayesianSingleTaskGP(train_X=train_X, train_Y=train_Y)
            nuts_options = {
                "warmup_steps": 512,
                "num_samples": 256,
                "thinning": 16
            }
            nuts_options.update((k,v) for k,v in kwargs.items() if k in nuts_options)
            fit_fully_bayesian_model_nuts(
                gp, 
                disable_progbar=True,
                **nuts_options
            )

            lengthscales = gp.median_lengthscale.detach()

        # get sorted dim indices by lengthscale in ascending order
        # if lengthscales[i] is the nth smallest entry, then dims_ordered[n] = i
        dims_ordered = torch.argsort(lengthscales)

        # get ranking of lengthscales by dim indices in ascending order
        # if lengthscales[i] is the nth smallest entry, then ranking[i] = n
        ranking = torch.argsort(torch.argsort(lengthscales))

        res[datasize] = {
            "lengthscales": lengthscales, 
            "dims_ordered": dims_ordered,
            "ranking": ranking
        }

        if verbose:
            print("lengthscales: ", lengthscales)
            print("dims_ordered: ", dims_ordered)
            
        if datasize > train_X.shape[0]:
            break

    return res




if __name__ == "__main__":

    base_problem = Hartmann()

    emb_problem = EmbeddedTestProblem(
        input_dim=50, base_problem=base_problem, seed=0
    )

    print("true indices of important dims: ", emb_problem.emb_indices)

    oracle_init_X, _ = oracle(emb_problem, perturb_option="ub")
    oracle_init_X = oracle_init_X.to(torch.double)

    saasgp(
        problem=emb_problem,
        n_samples=50,
        # inference_method="mle", # incredibly slow, don't know why
        inference_method="nuts", # also incredibly slow, don't know why
        checkpoints=[0.5, 1],
        verbose=True,
        # init_X=oracle_init_X
    )