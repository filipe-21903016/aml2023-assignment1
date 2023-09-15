import numpy as np
import sklearn.gaussian_process
import sklearn.gaussian_process.kernels
import typing

from scipy.stats import norm


class SequentialModelBasedOptimization(object):

    def __init__(self):
        """
        Initializes empty variables for the model, the list of runs (capital R), and the incumbent
        (theta_inc being the best found hyperparameters, theta_inc_performance being the performance
        associated with it)
        """
        self.model = sklearn.gaussian_process.GaussianProcessRegressor(
            kernel=sklearn.gaussian_process.kernels.Matern(), random_state=1)
        self.capital_r = None
        self.theta_inc_performance = None
        self.theta_inc = None

    def initialize(self, capital_phi: typing.List[typing.Tuple[np.array, float]]) -> None:
        """
        Initializes the model with a set of initial configurations, before it can make recommendations
        which configurations are in good regions. Note that we are maximizing (high values are preferred)

        :param capital_phi: a list of tuples, each tuple being a configuration and the performance (typically,
        accuracy)
        """
        self.capital_r = capital_phi
        for configuration, performance in capital_phi:
            if self.theta_inc_performance is None or performance > self.theta_inc_performance:
                self.theta_inc = configuration
                self.theta_inc_performance = performance

    def fit_model(self) -> None:
        """
        Fits the Gaussian Process model on the complete run list.
        """
        configurations = [theta[0] for theta in self.capital_r]
        performances = [theta[1] for theta in self.capital_r]
        self.model.fit(configurations, performances)

    def select_configuration(self, capital_theta: np.array) -> np.array:
        """
        Determines which configurations are good, based on the internal Gaussian Process model.
        Note that we are maximizing (high values are preferred)

        :param capital_theta: A (n, m) array, each column represents a hyperparameter and each row
        represents a configuration
        :return: A size n vector, same size as each element representing the EI of a given
        configuration
        """
        ei = self.expected_improvement(self.model, self.theta_inc_performance, capital_theta)
        # TODO: ei now contains for each element in capital_theta the expected improvement
        # return the element in capital_theta with the highest expected improvement
        raise NotImplementedError()

    @staticmethod
    def expected_improvement(model: sklearn.gaussian_process.GaussianProcessRegressor,
                             f_star: float, theta: np.array) -> np.array:
        """
        Acquisition function that determines which configurations are good and which
        are not good.

        :param model: The internal Gaussian Process model (should be fitted already)
        :param f_star: The current incumbent (theta_inc)
        :param theta: A (n, m) array, each column represents a hyperparameter and each row
        represents a configuration
        :return: A size n vector, same size as each element representing the EI of a given
        configuration
        """
        # TODO: see slides lecture 2
        raise NotImplementedError()

    def update_runs(self, run: typing.Tuple[np.array, float]):
        """
        After a configuration has been selected and ran, it will be added to run list
        (so that the model can be trained on it during the next iterations)
        Note that this is a extremely simplified version of the intensify function.
        Intensify can only be used when working across multiple random seeds, cross-
        validation folds, etc

        :param run: A 1D vector, each element represents a hyperparameter
        """
        self.capital_r.append(run)
        # TODO: update theta_inc and theta_performance, if needed
        raise NotImplementedError()
