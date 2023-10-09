import numpy as np
import sklearn.gaussian_process
import sklearn.gaussian_process.kernels
import typing

from scipy.stats import norm

from matplotlib import pyplot as plt


from warnings import catch_warnings
from warnings import simplefilter


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
        #New
        self.best_gaussian = []
        self.gaussian_scores = []
        self.best_gaussian_scores = []

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
        with catch_warnings():
            simplefilter("ignore")
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
        return capital_theta[np.argmax(ei)]

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
        mu_values, sigma_values = model.predict(theta, return_std=True)
        ei_values = np.array([])
        for i in range(len(mu_values)):
            mu = mu_values[i]
            sigma = sigma_values[i]
            z = (mu - f_star)/sigma
            ei_values = np.append(ei_values, (mu - f_star) * norm.cdf(z) + sigma * norm.pdf(z))
        
        return ei_values

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
        configuration = run[0]
        performance = run[1]
        if performance > self.theta_inc_performance:
            self.theta_inc = configuration
            self.theta_inc_performance = performance
            self.best_gaussian.append(configuration)
            self.best_gaussian_scores.append(performance)
        else:
            self.best_gaussian.append(self.theta_inc)
            self.best_gaussian_scores.append(self.theta_inc_performance)

        self.gaussian_scores.append(performance)

        return None
    
    #NEW
    def return_best_configuration(self):
        return self.theta_inc_performance, self.theta_inc
    
    """def plot_gaussian(self):
        X = []
        for i in self.capital_r:
            X.append(i[0])
        ysamples, _ = self.model.predict(X, return_std=True)
        plt.plot(ysamples)
        plt.ylim(0, 1)
        plt.show()

    def plot_best_gaussian(self):
        ysamples = self.model.predict(self.best_gaussian)
        unique_values, unique_indices = np.unique(ysamples, return_inverse=True)
        colormap = plt.get_cmap("viridis")
        num_colors = len(unique_values)
        colors = [colormap(i / num_colors) for i in range(num_colors)]
        fig, ax = plt.subplots()
        for i in range(num_colors):
            group_indices = np.where(unique_indices == i)[0]
            ax.plot(group_indices, ysamples[group_indices], color=colors[i])

        plt.ylim(0, 1)
        plt.show()"""
    
    def plot_gaussian_scores(self):
        plt.plot(self.gaussian_scores)
        plt.ylim(0, 1)
        plt.show()

    def plot_best_gaussian_scores(self):
        scores = np.array(self.best_gaussian_scores)
        unique_values, unique_indices = np.unique(scores, return_inverse=True)
        colormap = plt.get_cmap("viridis")
        num_colors = len(unique_values)
        colors = [colormap(i / num_colors) for i in range(num_colors)]
        fig, ax = plt.subplots()
        for i in range(num_colors):
            group_indices = np.where(unique_indices == i)[0].astype(int)  # Convert to integer
            ax.plot(group_indices, scores[group_indices], color=colors[i])

        plt.ylim(0, 1)
        plt.show()

