"""
  For finite fidelity methods.
"""

import numpy as np
from copy import deepcopy
from mf.mf_func import MFOptFunction, NoisyMFOptFunction


def get_finite_mfof_from_mfof(mfof, finite_fidels, is_normalised):
  """ Fidelities. """
  fmfof = deepcopy(mfof)
  fmfof.set_finite_fidels(finite_fidels, is_normalised)
  return fmfof


def mf_sko_fidel_chooser_single(mfof, *args, **kwargs):
  """ Fidelity chooser for mf_sko. """
  # TODO: just placeholder code. Fix this!
  if np.random.random() < 0.25:
    ret = mfof.opt_fidel
  else:
    cand_fidels = mfof.get_candidate_fidelities()
    ret = cand_fidels[np.random.randint(len(cand_fidels))]
  return ret

