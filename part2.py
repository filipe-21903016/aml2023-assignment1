
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

from assignment import SequentialModelBasedOptimization

import warnings
warnings.filterwarnings("ignore")



np.random.seed(0)



def stock_tuning(X_train, y_train, X_test, y_test):
    clf = svm.SVC()
    clf.fit(X_train, y_train)

    print("No hyperparameter tuning:")
    print(clf.score(X_test, y_test))



def random_search(X_train, y_train, X_test, y_test):
    clf = svm.SVC()
    param_dist = {
        "kernel": ['rbf'],
        "gamma": stats.uniform(1e-1, 1e-5),
        "C": stats.uniform(1, 1000),
    }

    n_iter_search = 100
    random_search = RandomizedSearchCV(
        clf, param_distributions=param_dist, n_iter=n_iter_search
    )

    start = time()
    random_search.fit(X_train, y_train)
    print(
        "RandomizedSearchCV took %.2f seconds for %d candidates parameter settings."
        % ((time() - start), n_iter_search)
    )

    print(random_search.score(X_test, y_test))



def grid_search(X_train, y_train, X_test, y_test):
    clf = svm.SVC()
    param_grid = {
        "kernel": ['rbf'],
        "gamma": [1e-1, 1e-5],
        "C": range(1,1000,20),
    }

    grid_search = GridSearchCV(clf, param_grid=param_grid)
    start = time()
    grid_search.fit(X_train, y_train)

    print(
        "GridSearchCV took %.2f seconds for %d candidate parameter settings."
        % (time() - start, len(grid_search.cv_results_["params"]))
    )

    print(grid_search.score(X_test, y_test))




def smbo(X_train, y_train, X_test, y_test):

    def optimizee(gamma, C):
        clf = svm.SVC()
        clf.set_params(kernel='rbf', gamma=gamma, C=C)
        clf.fit(X_train, y_train)
        return sklearn.metrics.accuracy_score(y_test, clf.predict(X_test))

    def sample_configurations(n_configurations):
        # function uses the ConfigSpace package, as developed at Freiburg University.
        # most of this functionality can also be achieved by the scipy package
        # same hyperparameter configuration as in scikit-learn
        cs = ConfigSpace.ConfigurationSpace('sklearn.svm.SVC', 1)

        C = ConfigSpace.UniformFloatHyperparameter(
            name='C', lower=1, upper=1000, log=True, default_value=1.0)
        gamma = ConfigSpace.UniformFloatHyperparameter(
            name='gamma', lower=1e-05, upper=1, log=True, default_value=0.1)
        cs.add_hyperparameters([C, gamma])

        return np.array([(configuration['gamma'],
                            configuration['C'])
                        for configuration in cs.sample_configuration(n_configurations)])

    def sample_initial_configurations(n: int) -> typing.List[typing.Tuple[np.array, float]]:
        configs = sample_configurations(n)
        return [((gamma, C), optimizee(gamma, C)) for gamma, C in configs]

    start = time()

    smbo = SequentialModelBasedOptimization()
    smbo.initialize(sample_initial_configurations(10))

    for idx in range(100):
        if idx%50 == 0:
            print('iteration %d/100' % idx)
        smbo.fit_model()
        theta_new = smbo.select_configuration(sample_configurations(64))
        performance = optimizee(theta_new[0], theta_new[1])
        smbo.update_runs((theta_new, performance))

    print(
        "SMBO took %.2f seconds for 100 iterations."
        % (time() - start)
    )
    best_score, best_hp = smbo.return_best_configuration()


    clf = sklearn.svm.SVC()
    clf.set_params(kernel='rbf', gamma=best_hp[0], C=best_hp[1])
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))



def make_comparisons(bunch_dataset):
    X = pd.DataFrame(data= bunch_dataset.data, columns=bunch_dataset.feature_names)  
    y = bunch_dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)

    print('\n')
    stock_tuning(X_train, y_train, X_test, y_test)
    print('\n')
    random_search(X_train, y_train, X_test, y_test)
    print('\n')
    grid_search(X_train, y_train, X_test, y_test)
    print('\n')
    smbo(X_train, y_train, X_test, y_test)
    print('\n')

bunch_dataset = fetch_openml(data_id=1464, as_frame=True, parser="pandas")
bunch_dataset2 = fetch_openml(data_id=1494, as_frame=True, parser="pandas")


make_comparisons(bunch_dataset)