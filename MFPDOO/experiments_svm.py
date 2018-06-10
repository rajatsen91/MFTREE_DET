##### Code to run synthetic experiments#############
####Contact: rajat.sen@utexas.edu ##################

from MFPDOO_real import *

FIDEL_BOUNDS = [[100,5000]]   ##### [100,5000] samples in the fidelity bound. Ignore [99,100] as of now. 
MBUDGET = 360 ### Total budget in seconds
if __name__ == '__main__':
    fidel_bounds = np.array(FIDEL_BOUNDS)
    mfobject = NGMFOptFunction(fidel_bounds)
    nu_max = 1.0
    rho_max = 0.9
    print 'calculating time......'
    t1 = time.time()
    x = mfobject.eval_at_fidel_single_point_normalised([1.0,1.0],[0.5,0.5])
    t2 = time.time()
    print 'Time Taken at Highest Fidelity: ' + str(t2 - t1)
    t0 = t2 - t1
    l = int(MBUDGET/t0) + 1

    total_budget_mults = np.array([l])

    K = 2
    C_init = 0.8
    tol = 1e-3
    Debug = True
    POO_mult = 0.5


    run_experiment(mfobject,nu_max,rho_max,total_budget_mults,K,C_init,tol,POO_mult,Debug, image_name = './examples/results/svm_deterministic_new',isfid1 = False, t0 = t0,auto = True)