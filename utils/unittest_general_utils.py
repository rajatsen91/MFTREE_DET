"""
  Test cases for functions in general_utils.py
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import

import numpy as np
import general_utils
from base_test_class import BaseTestClass, execute_tests


class GeneralUtilsTestCase(BaseTestClass):
  """Unit test class for general utilities. """

  def __init__(self, *args, **kwargs):
    super(GeneralUtilsTestCase, self).__init__(*args, **kwargs)

  def setUp(self):
    """ Sets up attributes. """
    # For dictionary
    self.dict_1 = {'a':1, 'b':2, 'c':3}
    self.dict_2 = {'b':2, 'a':1, 'c':3}
    self.dict_3 = {'a':4, 'b':5, 'c':6}
    self.dict_4 = {'d':1, 'e':2, 'f':3}
    # For dist squared
    self.X1 = np.array([[1, 2, 3], [1, 2, 4], [2, 3, 4.5]])
    self.X2 = np.array([[1, 2, 4], [1, 2, 5], [2, 3, 5]])
    self.true_dist_sq = np.array([[1, 4, 6], [0, 1, 3], [2.25, 2.25, 0.25]])

  def test_compare_dict(self):
    """ Test compare_dict function."""
    self.report('dicts_are_equal')
    assert general_utils.dicts_are_equal(self.dict_1, self.dict_2)
    assert not general_utils.dicts_are_equal(self.dict_1, self.dict_3)
    assert not general_utils.dicts_are_equal(self.dict_1, self.dict_4)

  def test_dist_squared(self):
    """ Tests the squared distance function. """
    self.report('dist_squared')
    comp_dist_sq = general_utils.dist_squared(self.X1, self.X2)
    assert (self.true_dist_sq == comp_dist_sq).all()

  def test_mapping_to_cube_and_bound(self):
    """ Test map_to_cube and map_to_bounds. """
    self.report('map_to_cube and map_to_bounds')
    bounds = np.array([[1, 3], [2, 4], [5, 6]])
    x = np.array([1.7, 3.1, 5.5])
    X = np.array([[1.7, 3.1, 5.5], [2.1, 2.9, 5.0]])
    y = np.array([0.35, 0.55, 0.5])
    Y = np.array([[0.35, 0.55, 0.5], [0.55, 0.45, 0]])
    # Map to cube
    y_ = general_utils.map_to_cube(x, bounds)
    Y_ = general_utils.map_to_cube(X, bounds)
    # Map to Bounds
    x_ = general_utils.map_to_bounds(y, bounds)
    X_ = general_utils.map_to_bounds(Y, bounds)
    # Check if correct.
    assert np.linalg.norm(y - y_) < 1e-5
    assert np.linalg.norm(Y - Y_) < 1e-5
    assert np.linalg.norm(x - x_) < 1e-5
    assert np.linalg.norm(X - X_) < 1e-5

  def test_compute_average_sq_prediction_error(self):
    """ Tests compute_average_sq_prediction_error. """
    self.report('compute_average_sq_prediction_error')
    Y1 = [0, 1, 2]
    Y2 = [2, 0, 1]
    res = general_utils.compute_average_sq_prediction_error(Y1, Y2)
    assert np.abs(res - 2.0) < 1e-5

  def test_stable_cholesky(self):
    """ Tests for stable cholesky. """
    self.report('stable_cholesky')
    M = np.random.normal(size=(5, 5))
    M = M.dot(M.T)
    L = general_utils.stable_cholesky(M)
    assert np.linalg.norm(L.dot(L.T) - M) < 1e-5

  def test_draw_gaussian_samples(self):
    """ Tests for draw gaussian samples. """
    self.report('draw_gaussian_samples. Probabilistic test, could fail at times')
    num_samples = 10000
    num_pts = 3
    mu = list(range(num_pts))
    K = np.random.normal(size=(num_pts, num_pts))
    K = K.dot(K.T)
    samples = general_utils.draw_gaussian_samples(num_samples, mu, K)
    sample_mean = samples.mean(axis=0)
    sample_centralised = samples - sample_mean
    sample_covar = sample_centralised.T.dot(sample_centralised) / num_samples
    mean_tol = 4 * np.linalg.norm(mu) / np.sqrt(num_samples)
    covar_tol = 4 * np.linalg.norm(K) / np.sqrt(num_samples)
    mean_err = np.linalg.norm(mu - sample_mean)
    covar_err = np.linalg.norm(K - sample_covar)
    self.report('Mean error (tol): ' + str(mean_err) + ' (' + str(mean_tol) + ')',
                'test_result')
    self.report('Cov error (tol): ' + str(covar_err) + ' (' + str(covar_tol) + ')',
                'test_result')
    assert mean_err < mean_tol
    assert covar_err < covar_tol


if __name__ == '__main__':
  execute_tests()

