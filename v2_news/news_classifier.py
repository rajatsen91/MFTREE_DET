"""
  20 News groups
"""


from copy import deepcopy
import numpy as np
from scipy.stats import norm
from scipy import integrate
# Local imports
from mf.mf_func import MFOptFunction
# For this
#from v2_news.news_classifier import get_kfold_val_score
#from v2_news import util
import sklearn
import warnings
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

NUM_FOLDS = 5
NUM_JOBS = 5    ##### should be ideally equal to number of processors in your machine
problem_bounds_raw = [[1e-2, 1e3], [1e-2, 1e3]] ########## Domain X for parameter tuning ########################



class NGMFOptFunction(MFOptFunction):
  """ MFOptFunction for SN data. """

  def __init__(self, fidel_bounds):
    """ Constructor. """
    self._load_data2()
    self.max_data = fidel_bounds[0,1]
    mf_func = self._mf_func
    fidel_cost_func = self._fidel_cost
    domain_bounds = np.log(np.array(problem_bounds_raw))
    opt_fidel_unnormalised = np.array([self.max_data])
    super(NGMFOptFunction, self).__init__(mf_func, fidel_cost_func, fidel_bounds,
                                          domain_bounds, opt_fidel_unnormalised,
                                          vectorised=False)

  def _fidel_cost(self, z):
    """ cost function """
    return 0.01 + (z[0]/self.max_data)

  def _mf_func(self, z, x):
    """ The MF Func. """
    clf = SVC(C=np.exp(x[0]), kernel='rbf', gamma=np.exp(x[1]),
                          tol=1e-20, max_iter=100)
    num_data_curr = int(z[0])
    feat_curr = self.features[1:num_data_curr]
    label_curr = self.labels[1:num_data_curr]
    return get_kfold_val_score(clf, feat_curr, label_curr)

  def _load_data(self):
    """ loads the data. """
    print('Loading data ...')
    data_path = './news/data/'
    files = sklearn.datasets.load_files(data_path)
    word_counts = util.bagOfWords(files.data)
    tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=True).fit(
                                                                              word_counts)
    features = tf_transformer.transform(word_counts)
    labels = files.target
    self.features, self.labels = deterministic_shuffle(features, labels)
    print('Done!')


  def _load_data2(self):
    print('Loading data ...')
    newsgroups_train = fetch_20newsgroups(subset='all')
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(newsgroups_train.data)
    labels = newsgroups_train.target
    self.features,self.labels = deterministic_shuffle(features,labels)
    print('Done!')






def get_kfold_val_score2(clf, X, Y, num_folds=None):
  if num_folds is None:
    num_folds = NUM_FOLDS
  max_folds = 5.0
  num_folds = int(min(max_folds, num_folds))
#   X, Y = deterministic_shuffle(X, Y)
  data_size = len(Y)
  acc_vals = []
  for i in range(num_folds):
    val_start_idx = int(i * data_size / max_folds)
    val_end_idx = int((i+1) * data_size / max_folds)
    val_idxs = np.array(range(val_start_idx, val_end_idx))
    tr_idxs = np.array(range(0, val_start_idx) + range(val_end_idx, data_size))
    X_tr = X[tr_idxs]
    Y_tr = Y[tr_idxs]
    X_va = X[val_idxs]
    Y_va = Y[val_idxs]
    acc_vals.append(get_val_score(clf, X_tr, Y_tr, X_va, Y_va))
  # Return mean
  return np.array(acc_vals).mean()


def get_kfold_val_score(clf, X, Y, num_folds=None,random_seed = 512):
  st0 = np.random.get_state()
  np.random.seed(random_seed)
  if num_folds is None:
    num_folds = NUM_FOLDS
  max_folds = 5.0
  num_folds = int(min(max_folds, num_folds))
  Kf = KFold(n_splits = num_folds, shuffle = True, random_state = random_seed)
  acc = cross_val_score(clf,X = X,y = Y,cv=Kf,n_jobs=NUM_JOBS,scoring='accuracy')
  np.random.set_state(st0)
  return acc.mean()

def deterministic_shuffle(X, Y, random_seed=512):
  """ deterministically shuffles. """
  st0 = np.random.get_state()
  np.random.seed()
  idxs = np.random.permutation(len(Y))
  X = X[idxs]
  Y = Y[idxs]
  np.random.set_state(st0)
  return X, Y

def get_val_score(clf, X_tr, Y_tr, X_va, Y_va):
  """ Computes score on validation set. """
  with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    clf.fit(X_tr, Y_tr)
    Y_pred = clf.predict(X_va)
    acc = (Y_pred == Y_va).mean()
    return acc
