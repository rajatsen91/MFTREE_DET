##### Code to run synthetic experiments#############
####Contact: rajat.sen@utexas.edu ##################

from MFPDOO_synthetic import *

if __name__ == '__main__':
    hartman_3d = get_mf_hartmann_as_mfof(1,3)
    hartman_6d = get_mf_hartmann_as_mfof(1,6)
    branin = get_mf_branin_as_mfof(1)
    curin = get_mf_currin_exp_as_mfof()
    borehole = get_mf_borehole_as_mfof()

    mfobject = borehole
    nu_max = 2.0
    rho_max = 0.95
    total_budget_mults = np.arange(10,210,10)
    K = 2
    C_init = 0.0001
    tol = 1e-3
    Debug = True
    POO_mult = 0.1

    run_experiment(mfobject,nu_max,rho_max,total_budget_mults,K,C_init,tol,POO_mult,Debug, image_name = './examples/results/borehole_all_deterministic',isExact =  True, isfid1 = True, isAll = True)
