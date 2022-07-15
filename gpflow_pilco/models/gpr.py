#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import gpflow
import tensorflow as tf
from gpflow.kernels import (Kernel,
                            SquaredExponential,
                            SeparateIndependent,
                            LinearCoregionalization)
from gpflow_pilco.models import MultivariateGPR
from gpflow_pilco.models import initializers
# from gpflow_pilco.models.core import CallableModel, LabelledData
from gpflow_pilco.models.core import LabelledData
from gpflow_pilco.models.mean_functions import Constant, Zero
from gpflow_sampling import models as pathwise_models
from typing import Callable

# ---- Exports
__all__ = ("GPR", "PathwiseGPR")


# ==============================================
#                                            gpr
# ==============================================
class GPR(Callable, MultivariateGPR):
  def __init__(self, *args, prior: Callable = None, **kwargs):
    super().__init__(*args, **kwargs)
    self.prior = prior

  def __call__(self, x: tf.Tensor, **kwargs):
    return self.predict_f(x, **kwargs)

  def maximum_log_likelihood_objective(self):
    objective = self.log_marginal_likelihood()
    if self.prior is not None:
      objective += self.prior(self)
    return objective

  @classmethod
  def initialize(cls,
                 data: LabelledData,
                 mean_function: Callable = "default",
                 **kwargs):

    X, Y = data
    output_dim = Y.shape[-1]

    # Define custom mean function, allows for broadcasting
    if mean_function == "default":
      mean_function = Constant(c=output_dim * [0])
    elif mean_function == "zero":
      # Overwrite the mean function to the Zero function
      mean_function = Zero(output_dim=output_dim)




    kernels = []
    for i in range(output_dim):
      kern = SquaredExponential()
      kern.lengthscales = initializers.lengthscales_median(X)
      kernels.append(kern)

    kernel = SeparateIndependent(kernels=kernels)

    return cls(kernel=kernel, data=data, mean_function=mean_function, **kwargs)


class PathwiseGPR(pathwise_models.PathwiseGPR, GPR):
  def __init__(self, *args, prior: Callable = None, **kwargs):
      pathwise_models.PathwiseGPR.__init__(self, *args, **kwargs)
      self.prior = prior

  def __call__(self, x: tf.Tensor, **kwargs):
    return self.predict_f_samples(x, **kwargs)
