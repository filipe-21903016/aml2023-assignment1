import numpy as np

import pandas as pd

import typing

import sklearn.gaussian_process
import sklearn.gaussian_process.kernels
from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier, MLPRegressor

import scipy.stats as stats
from scipy.stats import norm

from matplotlib import pyplot as plt

from time import time
import ConfigSpace

import sklearn.model_selection

import warnings
warnings.filterwarnings("ignore")
from warnings import catch_warnings
from warnings import simplefilter

OUTPUT_DIR = "./outputs/"

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
    
    def return_best_configuration(self):
        return self.theta_inc_performance, self.theta_inc
    

def stock_tuning(X_train, y_train, model):
    start = time()
    model.fit(X_train, y_train)
    return model, time()-start

def random_search(X_train, y_train, model, param_dist, n_iter):
    random_search = RandomizedSearchCV(
        model, param_distributions=param_dist, n_iter=n_iter)
    start = time()
    random_search.fit(X_train, y_train)
    return random_search, time()-start

def grid_search(X_train, y_train, model, param_grid):
    grid_search = GridSearchCV(model, param_grid=param_grid)
    start = time()
    grid_search.fit(X_train, y_train)
    return grid_search, time()-start

def smbo_svm(X_train, y_train, X_test, y_test, n_iter, mode, plot=False):
    def optimizee(param1, param2):
        if mode == 'SVC':
            clf = svm.SVC()
            clf.set_params(kernel='rbf', gamma=param1, C=param2)
        elif mode == 'SVR':
            clf = svm.SVR()
            clf.set_params(kernel='rbf', epsilon=param1, C=param2)
        elif mode == 'MLPC':
            clf = MLPClassifier(random_state=1, max_iter=200,
                                activation='relu', solver='sgd', learning_rate='constant')
            clf.set_params(alpha=param1, learning_rate_init=param2)
        elif mode == 'MLPR':
            clf = MLPRegressor(random_state=1, max_iter=200,
                               activation='relu', solver='sgd', learning_rate='constant')
            clf.set_params(alpha=param1, learning_rate_init=param2)
        clf.fit(X_train, y_train)
        return clf.score(X_test, y_test)

    def sample_configurations_svc(n_configurations):
        cs = ConfigSpace.ConfigurationSpace('sklearn.svm.SVC', 1)

        C = ConfigSpace.UniformFloatHyperparameter(
            name='C', lower=1, upper=1000, log=True, default_value=1.0)
        gamma = ConfigSpace.UniformFloatHyperparameter(
            name='gamma', lower=1e-05, upper=1, log=True, default_value=0.1)
        cs.add_hyperparameters([C, gamma])

        return np.array([(configuration['gamma'],
                          configuration['C'])
                        for configuration in cs.sample_configuration(n_configurations)])

    def sample_configurations_svr(n_configurations):
        cs = ConfigSpace.ConfigurationSpace('sklearn.svm.SVR', 1)

        C = ConfigSpace.UniformFloatHyperparameter(
            name='C', lower=1, upper=1000, log=True, default_value=1.0)
        epsilon = ConfigSpace.UniformFloatHyperparameter(
            name='epsilon', lower=1e-05, upper=1, log=True, default_value=0.1)
        cs.add_hyperparameters([C, epsilon])

        return np.array([(configuration['epsilon'],
                          configuration['C'])
                        for configuration in cs.sample_configuration(n_configurations)])

    def sample_configurations_mlpc(n_configurations):
        cs = ConfigSpace.ConfigurationSpace(
            'sklearn.neural_network.MLPClassifier', 1)

        alpha = ConfigSpace.UniformFloatHyperparameter(
            name='alpha', lower=1e-05, upper=1, log=True, default_value=0.0001)
        learning_rate_init = ConfigSpace.UniformFloatHyperparameter(
            name='learning_rate_init', lower=1e-05, upper=1e-1, log=True, default_value=0.001)
        cs.add_hyperparameters([alpha, learning_rate_init])

        return np.array([(configuration['alpha'],
                          configuration['learning_rate_init'])
                        for configuration in cs.sample_configuration(n_configurations)])

    def sample_initial_configurations(n: int) -> typing.List[typing.Tuple[np.array, float]]:
        if mode == 'SVC':
            configs = sample_configurations_svc(n)
            return [((gamma, C), optimizee(gamma, C)) for gamma, C in configs]
        elif mode == 'SVR':
            configs = sample_configurations_svr(n)
            return [((epsilon, C), optimizee(epsilon, C)) for epsilon, C in configs]
        elif mode in ['MLPC', 'MLPR']:
            configs = sample_configurations_mlpc(n)
            return [((alpha, learning_rate_init), optimizee(alpha, learning_rate_init)) for alpha, learning_rate_init in configs]

    start = time()

    smbo = SequentialModelBasedOptimization()
    smbo.initialize(sample_initial_configurations(10))

    for _ in range(n_iter):
        smbo.fit_model()
        if mode == 'SVC':
            theta_new = smbo.select_configuration(
                sample_configurations_svc(64))
        elif mode == 'SVR':
            theta_new = smbo.select_configuration(
                sample_configurations_svr(64))
        elif mode in ['MLPC', 'MLPR']:
            theta_new = smbo.select_configuration(
                sample_configurations_mlpc(64))

        performance = optimizee(theta_new[0], theta_new[1])
        smbo.update_runs((theta_new, performance))

    end = time() - start

    _, best_hp = smbo.return_best_configuration()

    if mode == 'SVC':
        clf = sklearn.svm.SVC()
        clf.set_params(kernel='rbf', gamma=best_hp[0], C=best_hp[1])
    elif mode == 'SVR':
        clf = sklearn.svm.SVR()
        clf.set_params(kernel='rbf', epsilon=best_hp[0], C=best_hp[1])
    elif mode == 'MLPC':
        clf = MLPClassifier(random_state=1, max_iter=200,
                            activation='relu', solver='sgd', learning_rate='constant')
        clf.set_params(alpha=best_hp[0], learning_rate_init=best_hp[1])
    elif mode == 'MLPR':
        clf = MLPRegressor(random_state=1, max_iter=200,
                           activation='relu', solver='sgd', learning_rate='constant')
        clf.set_params(alpha=best_hp[0], learning_rate_init=best_hp[1])

    clf.fit(X_train, y_train)
    return clf, end, (smbo.gaussian_scores, smbo.best_gaussian_scores)

def acc_score(model, X_test, y_test):
    return model.score(X_test, y_test)

def make_comparisons(param_rand, param_grid, mode, dataset_ids, smbo_plot=False):
    results = pd.DataFrame(
        columns=["DatasetID", "Stock", "Grid", "Random", "SMBO", "Stock_Time",
                 "Grid_Time", "Random_Time", "SMBO_Time", "SMBO_Gaussian_Scores", "SMBO_Best_Gaussian_Scores"])

    for id_ in dataset_ids:
        dataset = fetch_openml(data_id=id_, as_frame=True, parser="pandas")
        X = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
        y = dataset.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=0, test_size=0.3)

        if mode == 'SVC':
            clf1, clf2, clf3 = svm.SVC(), svm.SVC(), svm.SVC()

        elif mode == 'SVR':
            clf1, clf2, clf3 = svm.SVR(), svm.SVR(), svm.SVR()

        elif mode == 'MLPC':
            clf1, clf2, clf3 = MLPClassifier(random_state=1, max_iter=200), MLPClassifier(
                random_state=1, max_iter=200), MLPClassifier(random_state=1, max_iter=200)

        elif mode == 'MLPR':
            clf1, clf2, clf3 = MLPRegressor(random_state=1, max_iter=200), MLPRegressor(
                random_state=1, max_iter=200), MLPRegressor(random_state=1, max_iter=200)

        stock_model, stock_time = stock_tuning(X_train, y_train, clf1)
        stock_acc = acc_score(stock_model, X_test, y_test)

        rand_model, rand_time = random_search(
            X_train, y_train, clf2, param_rand, 100)
        rand_acc = acc_score(rand_model, X_test, y_test)

        grid_model, grid_time = grid_search(X_train, y_train, clf3, param_grid)
        grid_acc = acc_score(grid_model, X_test, y_test)

        smbo_model, smbo_time, (gaussian_scores, best_gaussian_scores) = smbo_svm(
            X_train, y_train, X_test, y_test, 100, mode, smbo_plot)
        smbo_acc = acc_score(smbo_model, X_test, y_test)

        result = {
            "DatasetID": id_,
            "Stock": round(stock_acc, 2),
            "Grid": round(grid_acc, 2),
            "Random": round(rand_acc, 2),
            "SMBO": round(smbo_acc, 2),
            "Stock_Time": stock_time,
            "Grid_Time": grid_time,
            "Random_Time": rand_time,
            "SMBO_Time": smbo_time,
            "SMBO_Gaussian_Scores": gaussian_scores,
            "SMBO_Best_Gaussian_Scores": best_gaussian_scores
        }

        results = pd.concat(
            [results, pd.DataFrame([result])], ignore_index=True)
    return results

def save_results_df(results, filename):
    results.to_csv(f"{OUTPUT_DIR}{filename}.csv", sep=",", index=False)

def plot_accuracies(results, model, show=False, save=False, filename=None):
    plt.clf()
    for gaussian_scores in results['SMBO_Gaussian_Scores']:
        plt.plot(gaussian_scores)

    plt.ylim(0, 1)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend(results['DatasetID'])
    plt.title(f'{model} SMBO')
    if show:
        plt.show()
    if save:
        plt.savefig(f'{OUTPUT_DIR}{filename}.png')

def plot_best_accuracies(results, model, show=False, save=False, filename=None):
    plt.clf()
    for gaussian_scores in results['SMBO_Best_Gaussian_Scores']:
        plt.plot(gaussian_scores)

    plt.ylim(0, 1)
    plt.xlabel("Iteration")
    plt.ylabel("Best Accuracy")
    plt.legend(results['DatasetID'])
    plt.title(f'{model} SMBO')
    if show:
        plt.show()
    if save:
        plt.savefig(f'{OUTPUT_DIR}{filename}.png')

if __name__ == '__main__':
    classification_datasets = [1464, 1491, 1494, 1504, 1063]
    regression_datasets = [8, 560, 1090, 44223]
    
    # Make comparison and get results for SVC
    param_rand_svc = {
        "kernel": ['rbf'],
        "gamma": stats.uniform(1e-1, 1e-5),
        "C": stats.uniform(1, 1000)
    }
    param_grid_svc = {
        "kernel": ['rbf'],
        "gamma": [1e-1, 1e-5],
        "C": range(1, 1000, 20)
    }
    results_svc = make_comparisons(
        param_rand_svc, param_grid_svc, 'SVC', classification_datasets)
    # Draw & Save plots
    plot_accuracies(results_svc, "SVC", save=True, filename="accuracies_svc")
    plot_best_accuracies(results_svc, "SVC", save=True,
                        filename="best_accuracies_svc")
    # Save results as CSV
    save_results_df(results_svc, "results_svc")    

    # Make comparison and get results for SVR
    param_rand_svr = {
        "kernel": ['rbf'],
        "epsilon": stats.uniform(1, 1e-4),
        "C": stats.uniform(1, 1000)
    }
    param_grid_svr = {
        "kernel": ['rbf'],
        "epsilon": [1, 1e-4],
        "C": range(1, 1000, 20)
    }
    results_svr = make_comparisons(
        param_rand_svr, param_grid_svr, 'SVR', regression_datasets)
    # Draw & Save plots
    plot_accuracies(results_svr, "SVR", save=True,
                    filename="accuracies_svr")
    plot_best_accuracies(results_svr, "SVR", save=True,
                        filename="best_accuracies_svr")
    # Save results as CSV
    save_results_df(results_svr, "results_svr")


    # Make comparison and get results for MLPC 
    param_rand_MLP = {
        "activation": ['relu'],
        "solver": ['sgd'],
        "learning_rate": ['constant'],
        "learning_rate_init": stats.uniform(1e-1, 1e-5),
        "alpha": stats.uniform(1e-1, 1e-5)
    }
    param_grid_MLP = {
        "activation": ['relu'],
        "solver": ['sgd'],
        "learning_rate": ['constant'],
        "learning_rate_init": [1e-1, 1e-5],
        "alpha": [1e-1, 1e-5]
    }
    results_mlpc = make_comparisons(
        param_rand_MLP, param_grid_MLP, 'MLPC', classification_datasets)
    # Draw & Save plots
    plot_accuracies(results_mlpc, "MLPC", save=True, filename="accuracies_mlpc")
    plot_best_accuracies(results_mlpc, "MLPC", save=True,
                        filename="best_accuracies_mlpc")
    # Save results as CSV
    save_results_df(results_mlpc, "results_mlpc")
    
    # Make comparison and get results for MLPR
    results_mlpr = make_comparisons(
        param_rand_MLP, param_grid_MLP, 'MLPR', regression_datasets)
    # Draw & Save plots
    plot_accuracies(results_mlpr, "MLPR", save=True, filename="accuracies_mlpr")
    plot_best_accuracies(results_mlpr, "MLPR", save=True,
                        filename="best_accuracies_mlpr")
    # Save results as CSV
    save_results_df(results_mlpr, "results_mlpr")