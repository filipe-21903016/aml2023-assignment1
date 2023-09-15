import ConfigSpace
import numpy as np
import sklearn.datasets
import sklearn.model_selection
import sklearn.svm
import typing
import unittest

from assignment import SequentialModelBasedOptimization


class TestMetaModels(unittest.TestCase):

    def test_branin_hoo(self):
        np.random.seed(0)

        def optimizee(x1: typing.Union[float, np.array],
                      x2: typing.Union[float, np.array]) -> typing.Union[float, np.array]:
            a = 1
            b = 5.1 / (4 * np.pi ** 2)
            c = 5 / np.pi
            r = 6
            s = 10
            t = 1 / (8 * np.pi)
            return (a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s) * -1

        def sample_configurations(n: int):
            x1 = np.random.uniform(-5, 10, (n, 1))
            x2 = np.random.uniform(0, 15, (n, 1))
            return np.concatenate((x1, x2), axis=1)

        def sample_initial_configurations(n: int) -> typing.List[typing.Tuple[np.array, float]]:
            configs = sample_configurations(n)
            return [(x, optimizee(x[0], x[1])) for x in configs]

        smbo = SequentialModelBasedOptimization()
        smbo.initialize(sample_initial_configurations(10))

        for idx in range(128):
            smbo.fit_model()
            theta_new = smbo.select_configuration(sample_configurations(8))
            performance = optimizee(theta_new[0], theta_new[1])
            smbo.update_runs((theta_new, performance))
        f_min_x = np.array([-np.pi, 12.275])
        f_min_y = optimizee(f_min_x[0], f_min_x[1])
        smbo.update_runs((f_min_x, f_min_y))
        self.assertAlmostEqual(f_min_y, smbo.theta_inc_performance)
        self.assertEqual(f_min_x[0], smbo.theta_inc[0])
        self.assertEqual(f_min_x[1], smbo.theta_inc[1])

    def test_optimize_svm(self):
        np.random.seed(0)

        data = sklearn.datasets.fetch_openml(name='iris', version=1)
        X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(
            data.data, data.target, test_size=0.33, random_state=1)

        def optimizee(gamma, C):
            clf = sklearn.svm.SVC()
            clf.set_params(kernel='rbf', gamma=gamma, C=C)
            clf.fit(X_train, y_train)
            return sklearn.metrics.accuracy_score(y_valid, clf.predict(X_valid))

        def sample_configurations(n_configurations):
            # function uses the ConfigSpace package, as developed at Freiburg University.
            # most of this functionality can also be achieved by the scipy package
            # same hyperparameter configuration as in scikit-learn
            cs = ConfigSpace.ConfigurationSpace('sklearn.svm.SVC', 1)

            C = ConfigSpace.UniformFloatHyperparameter(
                name='C', lower=0.03125, upper=32768, log=True, default_value=1.0)
            gamma = ConfigSpace.UniformFloatHyperparameter(
                name='gamma', lower=3.0517578125e-05, upper=8, log=True, default_value=0.1)
            cs.add_hyperparameters([C, gamma])

            return np.array([(configuration['gamma'],
                              configuration['C'])
                            for configuration in cs.sample_configuration(n_configurations)])

        def sample_initial_configurations(n: int) -> typing.List[typing.Tuple[np.array, float]]:
            configs = sample_configurations(n)
            return [((gamma, C), optimizee(gamma, C)) for gamma, C in configs]

        smbo = SequentialModelBasedOptimization()
        smbo.initialize(sample_initial_configurations(10))

        for idx in range(16):
            print('iteration %d/16' % idx)
            smbo.fit_model()
            theta_new = smbo.select_configuration(sample_configurations(64))
            performance = optimizee(theta_new[0], theta_new[1])
            smbo.update_runs((theta_new, performance))
