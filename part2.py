
import pandas as pd
import scipy.stats as stats
from time import time
import numpy as np
import typing
import ConfigSpace

import sklearn.model_selection
from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier, MLPRegressor

from matplotlib import pyplot as plt

from assignment import SequentialModelBasedOptimization

import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)


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
    if plot:
        smbo.plot_gaussian_scores()
        smbo.plot_best_gaussian_scores()
    return clf, end, (smbo.gaussian_scores, smbo.best_gaussian_scores)


def acc_score(model, X_test, y_test):
    return model.score(X_test, y_test)


def make_comparisons(param_rand, param_grid, mode, dataset_ids, smbo_plot=False):
    results = pd.DataFrame(
        columns=["DatasetID", "Stock", "Grid", "Random", "SMBO", "Stock_Time", "Grid_Time", "Random_Time", "SMBO_Time", "SMBO_Gaussian_Scores", "SMBO_Best_Gaussian_Scores"])

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
    cols = ["DatasetID", "Stock", "Grid", "Random", "SMBO",
            "Stock_Time", "Grid_Time", "Random_Time", "SMBO_Time"]
    results.to_csv(f"./outputs/{filename}.csv",
                   columns=cols, sep=",", index=False)


def plot_gaussian_scores(results, show=False, save=False, filename=None):
    plt.clf()
    for gaussian_scores in results['SMBO_Gaussian_Scores']:
        plt.plot(gaussian_scores)

    plt.ylim(0, 1)
    plt.xlabel("Iteration")
    plt.ylabel("Gaussian Score")
    plt.legend(results['DatasetID'])
    plt.title("SVC SMBO")
    if show:
        plt.show()
    if save:
        plt.savefig(f'outputs/{filename}.png')


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


# SVC - 1464, 1491, 1494, 1504, 1063
results_svc = make_comparisons(
    param_rand_svc, param_grid_svc, 'SVC', [1464, 1491, 1494, 1504, 1063])
# Plot Gaussian Scores
plot_gaussian_scores(results_svc, save=True, filename="gaussian_scores_svc")
# Save to CSV
save_results_df(results_svc, "results_svc")

# SVR
results_svr = make_comparisons(
    param_rand_svr, param_grid_svr, 'SVR', [8, 560])
# Plot Gaussian Scores
plot_gaussian_scores(results_svr, save=True, filename="gaussian_scores_svr")
# Save to CSV
save_results_df(results_svr, "results_svr")
