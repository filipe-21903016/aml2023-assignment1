
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
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter)
    start = time()
    random_search.fit(X_train, y_train)
    return random_search, time()-start
    

def grid_search(X_train, y_train, model, param_grid):
    grid_search = GridSearchCV(model, param_grid=param_grid)
    start = time()
    grid_search.fit(X_train, y_train)
    return grid_search, time()-start



def smbo_svm(X_train, y_train, X_test, y_test, n_iter, mode):

    def optimizee(param1, param2):
        if mode == 'SVC':
            clf = svm.SVC()
            clf.set_params(kernel='rbf', gamma=param1, C=param2)
        elif mode == 'SVR':
            clf = svm.SVR()
            clf.set_params(kernel='rbf', epsilon=param1, C=param2)
        elif mode == 'MLPC':
            clf = MLPClassifier(random_state=1, max_iter=200, activation = 'relu', solver = 'sgd', learning_rate = 'constant')
            clf.set_params(alpha=param1, learning_rate_init=param2)
        elif mode == 'MLPR':
            clf = MLPRegressor(random_state=1, max_iter=200, activation = 'relu', solver = 'sgd', learning_rate = 'constant')
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
        cs = ConfigSpace.ConfigurationSpace('sklearn.neural_network.MLPClassifier', 1)

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

    for idx in range(n_iter):
        smbo.fit_model()
        if mode == 'SVC':
            theta_new = smbo.select_configuration(sample_configurations_svc(64))
        elif mode == 'SVR':
            theta_new = smbo.select_configuration(sample_configurations_svr(64))
        elif mode in ['MLPC', 'MLPR']:
            theta_new = smbo.select_configuration(sample_configurations_mlpc(64))
       
        performance = optimizee(theta_new[0], theta_new[1])
        smbo.update_runs((theta_new, performance))

    end = time() - start


    best_score, best_hp = smbo.return_best_configuration()

    if mode == 'SVC':
        clf = sklearn.svm.SVC()
        clf.set_params(kernel='rbf', gamma=best_hp[0], C=best_hp[1])
    elif mode == 'SVR':
        clf = sklearn.svm.SVR()
        clf.set_params(kernel='rbf', epsilon=best_hp[0], C=best_hp[1])
    elif mode == 'MLPC':
        clf = MLPClassifier(random_state=1, max_iter=200, activation = 'relu', solver = 'sgd', learning_rate = 'constant')
        clf.set_params(alpha=best_hp[0], learning_rate_init=best_hp[1])
    elif mode == 'MLPR':
        clf = MLPRegressor(random_state=1, max_iter=200, activation = 'relu', solver = 'sgd', learning_rate = 'constant')
        clf.set_params(alpha=best_hp[0], learning_rate_init=best_hp[1])

    clf.fit(X_train, y_train)
    smbo.plot_gaussian_scores()
    smbo.plot_best_gaussian_scores()
    return clf, end



def acc_score(model, X_test, y_test):
    return model.score(X_test, y_test)


def make_comparisons(X, y, param_rand, param_grid, mode):

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)

    if mode == 'SVC':
        clf1, clf2, clf3 = svm.SVC(), svm.SVC(), svm.SVC()

    elif mode == 'SVR':
        clf1, clf2, clf3 = svm.SVR(), svm.SVR(), svm.SVR()

    elif mode == 'MLPC':
        clf1, clf2, clf3 = MLPClassifier(random_state=1, max_iter=200), MLPClassifier(random_state=1, max_iter=200), MLPClassifier(random_state=1, max_iter=200)

    elif mode == 'MLPR':
        clf1, clf2, clf3 = MLPRegressor(random_state=1, max_iter=200), MLPRegressor(random_state=1, max_iter=200), MLPRegressor(random_state=1, max_iter=200)
    
    m1, time1 = stock_tuning(X_train, y_train, clf1)
    print(f"\nStock - Accuracy: {acc_score(m1,X_test, y_test)}; Elapsed Time: {time1} seconds \n")
    m2, time2 = random_search(X_train, y_train, clf2, param_rand, 100)
    print(f"Random - Accuracy: {acc_score(m2,X_test, y_test)}; Elapsed Time: {time2} seconds \n")
    m3, time3 = grid_search(X_train, y_train, clf3, param_grid)
    print(f"Grid - Accuracy: {acc_score(m3,X_test, y_test)}; Elapsed Time: {time3} seconds \n")
    m4, time4 = smbo_svm(X_train, y_train, X_test, y_test,100, mode)
    print(f"SMBO - Accuracy: {acc_score(m4,X_test, y_test)}; Elapsed Time: {time4} seconds\n ")

bunch_dataset = fetch_openml(data_id=1464, as_frame=True, parser="pandas")
bunch_dataset2 = fetch_openml(data_id=1494, as_frame=True, parser="pandas")
bunch_dataset3 = fetch_openml(data_id=1504, as_frame=True, parser="pandas")
bunch_dataset4 = fetch_openml(data_id=1063, as_frame=True, parser="pandas")

bunch_dataset_svr = fetch_openml(data_id=8, as_frame=True, parser="pandas")
bunch_dataset_svr2 = fetch_openml(data_id=560, as_frame=True, parser="pandas")


X = pd.DataFrame(data= bunch_dataset.data, columns=bunch_dataset.feature_names)  
y = bunch_dataset.target

param_rand_svc = {
        "kernel": ['rbf'],
        "gamma": stats.uniform(1e-1, 1e-5),
        "C": stats.uniform(1, 1000)
    }

param_grid_svc = {
        "kernel": ['rbf'],
        "gamma": [1e-1, 1e-5],
        "C": range(1,1000,20)
    }

param_rand_svr = {
        "kernel": ['rbf'],
        "epsilon": stats.uniform(1, 1e-4),
        "C": stats.uniform(1, 1000)
    }

param_grid_svr = {
        "kernel": ['rbf'],
        "epsilon": [1, 1e-4],
        "C": range(1,1000,20)
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

make_comparisons(X, y, param_rand_svc, param_grid_svc, 'SVC')
