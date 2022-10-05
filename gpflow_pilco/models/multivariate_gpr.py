# From GPflow pull request: https://github.com/GPflow/GPflow/pull/1616

from typing import Optional, Tuple
import tensorflow as tf
import gpflow
from gpflow.kernels import Kernel
from gpflow.logdensities import multivariate_normal
from gpflow.mean_functions import MeanFunction
from gpflow.models.model import InputData, MeanAndVariance, RegressionData
from gpflow.models import GPModel
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor

from gpflow.config import default_jitter

class MultivariateGPR(GPModel, InternalDataTrainingLossMixin):
    r"""
    Multivariate extension of the vanilla Gaussian Process Regression.
    """
    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        mean_function: Optional[MeanFunction] = None,
        noise_variance: float = 1.0,
    ):
        likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        _, Y_data = data
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=Y_data.shape[-1])
        self.data = data_input_to_tensor(data)

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.log_marginal_likelihood()

    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.
        .. math::
            \log p(Y | \theta).
        """
        X, Y = self.data
        m = self.mean_function(X) # [N, 1]
        Kmm = self.kernel(X, full_cov = True, full_output_cov = True) # [M, P, M, P]

        # reshape to be compatible with multi output
        M, P = tf.shape(Y)[0], tf.shape(Y)[1]     
        Y = tf.reshape(Y, shape = (M * P, 1)) # [M*P, 1]
        m = tf.tile(m, (P, tf.constant(1))) # [M*P, 1]
        Kmm = tf.reshape(Kmm, shape = (M * P, M * P)) # [M*P, M*P]
        # add jitter to the diagional
        Kmm += default_jitter() * tf.eye(M * P, dtype=Kmm.dtype)
        if len(self.likelihood.variance.shape) > 0:
            Kmm += tf.linalg.diag(tf.repeat(self.likelihood.variance,M)) # [M*P, M*P]
        else:
            Kmm += tf.linalg.diag(tf.fill([M * P], self.likelihood.variance)) # [M*P, M*P]
        L = tf.linalg.cholesky(Kmm)  # [M*P, M*P]

        # [R,] log-likelihoods for each independent dimension of Y
        log_prob = multivariate_normal(Y, m, L)
        return tf.reduce_sum(log_prob)

    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        r"""
        This method computes predictions at X \in R^{N \x D} input points
        .. math::
            p(F* | Y)
        where F* are points on the GP at new data points, Y are noisy observations at training data points.
        """
        X_data, Y_data = self.data
        err = Y_data - self.mean_function(X_data)

        Kmm = self.kernel(X_data, full_cov = True, full_output_cov = True) # [M, P, M, P]
        Knn = self.kernel(Xnew, full_cov=full_cov, full_output_cov = full_output_cov) # [N, P, N, P] or [N, N, P] or [P, N, N] or [N, P] 
        Kmn = self.kernel(X_data, Xnew, full_cov = True, full_output_cov = True) # [M, P, N, P]

        M, P, N, _ = tf.unstack(tf.shape(Kmn), num=Kmn.shape.ndims, axis=0)
        Kmn = tf.reshape(Kmn, (M * P, N, P)) # [M*P, N, P]
        err = tf.reshape(err, shape = (M * P, 1)) # [M*P, 1]
        Kmm = tf.reshape(Kmm, (M * P, M * P)) # [M*P, M*P]
        if len(self.likelihood.variance.shape) > 0:
            Kmm += tf.linalg.diag(tf.repeat(self.likelihood.variance,M)) # [M*P, M*P]
        else:
            Kmm += tf.linalg.diag(tf.fill([M * P], self.likelihood.variance)) # [M*P, M*P]
        # add jitter to the diagional
        Kmm += default_jitter() * tf.eye(M * P, dtype=Kmm.dtype)

        M, N, P = tf.unstack(tf.shape(Kmn), num=Kmn.shape.ndims, axis=0)
        Lm = tf.linalg.cholesky(Kmm) # [M*P, M*P]
        Kmn = tf.reshape(Kmn, (M, N * P))  # [M*P, N*P]
        A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)  # [M*P, N*P]
        Ar = tf.reshape(A, (M, N, P)) # [M*P, N, P]

        # compute the covariance due to the conditioning
        if full_cov and full_output_cov:
            fvar = Knn - tf.tensordot(Ar, Ar, [[0], [0]])  # [N, P, N, P]
        elif full_cov and not full_output_cov:
            At = tf.transpose(Ar)  # [P, N, M*P]
            fvar = Knn - tf.linalg.matmul(At, At, transpose_b=True)  # [P, N, N]
        elif not full_cov and full_output_cov:
            At = tf.transpose(Ar, [1, 0, 2])  # [N, M*P, P]
            fvar = Knn - tf.linalg.matmul(At, At, transpose_a=True)  # [N, P, P]
        elif not full_cov and not full_output_cov:
            fvar = Knn - tf.reshape(tf.reduce_sum(tf.square(A), [0]), (N, P)) # [N, P]

        # another backsubstitution in the unwhitened case
        A = tf.linalg.triangular_solve(tf.linalg.adjoint(Lm), A, lower=False)  # [M*P, N*P]
        fmean_zero = tf.linalg.matmul(err, A, transpose_a=True)  # [1, N*P]
        fmean_zero = tf.reshape(fmean_zero, (N, P))  # [N, P]
        return fmean_zero + self.mean_function(Xnew), fvar

