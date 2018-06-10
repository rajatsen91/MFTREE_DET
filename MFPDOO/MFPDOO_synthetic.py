
##### Code Base for running MFPDOO on synthetic functions ###############
#### Contact: rajat.sen@utexas.edu


###### Essential Imports #######################

import numpy as np
import Queue
from mf.mf_func import MFOptFunction
from utils.general_utils import map_to_cube
import sys
import matplotlib.pyplot as plt
import matplotlib
from examples.synthetic_functions import *
from multiprocessing import Process
import brewer2mpl
import pandas as pd
import matplotlib.pyplot as plt

###################################################

class MF_node(object):
    ''' 
    Base object for the node of a tree
    '''
    def __init__(self,cell,value,fidel,upp_bound,height,dimension):
        '''cell: tuple'''
        self.cell = cell
        self.value = value
        self.fidelity = fidel
        self.upp_bound = upp_bound
        self.height = height
        self.dimension = dimension

    def __cmp__(self,other):
        return cmp(other.upp_bound,self.upp_bound)

class MFoptTree(object):
    def __init__(self,mfobject, nu_max, rho_max, total_budget, K, C_init,tol = 1e-3,Randomize = False):
        self.mfobject = mfobject
        self.nu_max = nu_max
        self.rho_max = rho_max
        self.total_budget = total_budget
        self.K = K
        self.C = C_init
        self.tol = tol
        self.value_dict = {}
        self.qcount = 0
        self.last_z_hist = []
        self.Randomize = Randomize

    def get_value(self,cell,fidel):
        '''cell: tuple'''
        self.qcount = self.qcount + 1
        x = np.array([(s[0]+s[1])/2.0 for s in list(cell)])
        return self.mfobject.eval_at_fidel_single_point_normalised([fidel], x)

    def querie(self,cell,height, rho, nu,dimension,option = 1):
        diam = nu*(rho**height)
        if option == 1:
            z = min(max(1 - diam/self.C,self.tol),1.0)
        else:
            z = 1.0
        if cell in self.value_dict:
            current = self.value_dict[cell]
            if abs(current.fidelity - z) <= self.tol:
                value = current.value
                cost = 0
            else:
                value = self.get_value(cell,z)
                self.last_z_hist = self.last_z_hist + [z]
                if abs(value - current.value) > self.C*abs(current.fidelity - z):
                    self.C = 2.0*self.C
                current.value = value
                current.fidelity = z
                self.value_dict[cell] = current
                cost = self.mfobject.eval_fidel_cost_single_point_normalised([z])
        else:
            value = self.get_value(cell,z)
            self.last_z_hist = self.last_z_hist + [z]
            bhi = diam + self.C*(1.0 - z) + value
            self.value_dict[cell] = MF_node(cell,value,z,bhi,height,dimension)
            cost = self.mfobject.eval_fidel_cost_single_point_normalised([z])

        bhi = diam + self.C*(1.0 - z) + value
        current_object = MF_node(cell,value,z,bhi,height,dimension)
        return current_object,cost

    def split_children(self,current,rho,nu,option = 1):
        pcell = list(current.cell)
        span = [abs(pcell[i][1] - pcell[i][0]) for i in range(len(pcell))]
        if self.Randomize:
            dimension = np.random.choice(range(len(pcell)))
        else:
            dimension = np.argmax(span)
        dd = len(pcell)
        if dimension == current.dimension:
            dimension = (current.dimension - 1)%dd
        cost = 0
        h = current.height + 1
        l = np.linspace(pcell[dimension][0],pcell[dimension][1],self.K+1)
        children = []
        for i in range(len(l)-1):
            cell = []
            for j in range(len(pcell)):
                if j != dimension:
                    cell = cell + [pcell[j]]
                else:
                    cell = cell + [(l[i],l[i+1])]
            cell = tuple(cell)
            child,c = self.querie(cell, h, rho, nu,dimension,option)
            children = children + [child]
            cost = cost + c

        return children, cost


    def run_DOO(self,budget,nu,rho):
        leaf_Q = Queue.PriorityQueue()
        d = self.mfobject.domain_dim
        cell = tuple([tuple([0,1]) for t in range(d)])
        height = 0
        cost = 0
        current,c = self.querie(cell, height, rho, nu,0)
        cost = cost + c
        leaf_Q.put(current)
        dict_of_points = {}
        while cost <= budget:
            current = leaf_Q.get()
            dict_of_points[current.cell] = {'val':current.value, 'fidel': current.fidelity, 'height':current.height}
            children,curr_cost = self.split_children(current, rho, nu)
            if current.cell == children[0].cell:
                break
            cost = cost + curr_cost
            for child in children:
                leaf_Q.put(child)


        while not leaf_Q.empty():
            c = leaf_Q.get()
            dict_of_points[c.cell] = {'val':c.value, 'fidel': c.fidelity, 'height':c.height}

        maxi = float(-sys.maxint - 1)
        point = 0
        maxh = 0
        val = 0
        fidel = 0
        for key in dict_of_points:
            c = dict_of_points[key]
            if c['val'] - self.C*(1.0 - c['fidel']) > maxi: #- nu*(rho**c.height) > maxi:
                maxi = c['val'] - self.C*(1.0 - c['fidel']) #- nu*(rho**c.height)
                val = c['val']
                fidel = c['fidel']
                point = np.array([(s[0]+s[1])/2 for s in key])
                maxh = c['height']



        return val,fidel,point,cost,maxh


    def run_DOO_fid1(self,budget,nu,rho):
        leaf_Q = Queue.PriorityQueue()
        d = self.mfobject.domain_dim
        cell = tuple([tuple([0,1]) for t in range(d)])
        height = 0
        cost = 0
        current,c = self.querie(cell, height, rho, nu,0,2)
        cost = cost + c
        leaf_Q.put(current)
        dict_of_points = {}
        while cost <= budget:
            current = leaf_Q.get()
            dict_of_points[current.cell] = {'val':current.value, 'fidel': current.fidelity, 'height':current.height}
            children,curr_cost = self.split_children(current, rho, nu,2)
            if current.cell == children[0].cell:
                break
            cost = cost + curr_cost
            for child in children:
                leaf_Q.put(child)


        while not leaf_Q.empty():
            c = leaf_Q.get()
            dict_of_points[c.cell] = {'val':c.value, 'fidel': c.fidelity, 'height':c.height}

        maxi = float(-sys.maxint - 1)
        point = 0
        maxh = 0
        val = 0
        fidel = 0
        for key in dict_of_points:
            c = dict_of_points[key]
            if c['val'] - self.C*(1.0 - c['fidel']) > maxi: #- nu*(rho**c.height) > maxi:
                maxi = c['val'] - self.C*(1.0 - c['fidel']) #- nu*(rho**c.height)
                val = c['val']
                fidel = c['fidel']
                point = np.array([(s[0]+s[1])/2 for s in key])
                maxh = c['height']



        return val,fidel,point,cost,maxh

    def run_PDOO(self,mult = 0.5,Debug = False):
        Dm = int(np.log(self.K)/np.log(1/self.rho_max))
        n = self.total_budget/self.mfobject.eval_fidel_cost_single_point_normalised(np.array([1.0]))
        N = int(mult*Dm*np.log(n/np.log(n)))
        if Debug:
            print 'Number of MFDOO instances: ' + str(N)
            print 'Total Budget: ' + str(self.total_budget)
        budget = self.total_budget/float(N)
        nu = self.nu_max
        total_cost = 0.0
        results = []
        for i in range(N):
            if Debug:
                print 'MFDOO #: ' + str(i) + '  ',
            rho = (self.rho_max)**(float(N)/(N-i))
            est,fidel,point,cost,h = self.run_DOO(budget,nu,rho)
            if Debug:
                print rho,est,h
            results = results + [(est,fidel,point,cost,h)]
            total_cost = total_cost + cost
        temp = [s[0] - self.C*(1 - s[1]) for s in results]
        index = np.argmax(temp)
        self.last_results = (temp[index],results[index][2],total_cost)
        if Debug:
            print 'Cost Used: ' + str(total_cost)
        return temp[index],results[index][2],total_cost

    def run_PDOO_exact(self,mult = 0.5,Debug = False):
        Dm = int(np.log(self.K)/np.log(1/self.rho_max))
        n = self.total_budget/self.mfobject.eval_fidel_cost_single_point_normalised(np.array([1.0]))
        N = int(mult*Dm*np.log(n/np.log(n)))
        if Debug:
            print 'Number of MFDOO instances: ' + str(N)
            print 'Total Budget: ' + str(self.total_budget)
        budget = (self.total_budget - N*self.mfobject.eval_fidel_cost_single_point_normalised(np.array([1.0])))/float(N)
        nu = self.nu_max
        total_cost = 0.0
        results = []
        for i in range(N):
            if Debug:
                print 'MFDOO #: ' + str(i) + '  ',
            rho = (self.rho_max)**(float(N)/(N-i))
            est,fidel,point,cost,h = self.run_DOO(budget,nu,rho)
            if Debug:
                print rho,est,h
            results = results + [(est,fidel,point,cost,h)]
            total_cost = total_cost + cost
        temp = [self.mfobject.eval_at_fidel_single_point_normalised([1.0], s[2]) for s in results]
        self.last_z_hist = self.last_z_hist + [1.0]*N
        index = np.argmax(temp)
        total_cost = total_cost + N*self.mfobject.eval_fidel_cost_single_point_normalised(np.array([1.0]))
        self.last_results = (temp[index],results[index][2],total_cost)
        if Debug:
            print 'Cost Used: ' + str(total_cost)
        return temp[index],results[index][2],total_cost


    def run_PDOO_fid1(self,mult = 0.5,Debug = False):
        Dm = int(np.log(self.K)/np.log(1/self.rho_max))
        n = self.total_budget/self.mfobject.eval_fidel_cost_single_point_normalised(np.array([1.0]))
        N = int(mult*Dm*np.log(n/np.log(n)))
        if Debug:
            print 'Number of MFDOO instances: ' + str(N)
            print 'Total Budget: ' + str(self.total_budget)
        budget = self.total_budget/float(N)
        nu = self.nu_max
        total_cost = 0.0
        results = []
        for i in range(N):
            if Debug:
                print 'SOO #: ' + str(i) + '  ',
            rho = (self.rho_max)**(float(N)/(N-i))
            est,fidel,point,cost,h = self.run_DOO_fid1(budget,nu,rho)
            if Debug:
                print rho,est,h
            results = results + [(est,fidel,point,cost,h)]
            total_cost = total_cost + cost
        temp = [s[0] - self.C*(1 - s[1]) for s in results]
        index = np.argmax(temp)
        self.last_results = (temp[index],results[index][2],total_cost)
        if Debug:
            print 'Cost Used: ' + str(total_cost)
        return temp[index],results[index][2],total_cost




def run_experiment(mfobject,nu_max,rho_max,total_budget_mults,K,C_init,tol,POO_mult,Debug = False,Randomize = False,isPlot = True, image_name = 'default_image',isExact = True,isfid1 = False,isAll = False,isSave = True):
    R = []
    C = []
    C1 = []
    R1 = []
    for m in total_budget_mults:
        total_budget = m*mfobject.eval_fidel_cost_single_point_normalised([1.0])
        if isAll:
            MF = MFoptTree(mfobject, nu_max, rho_max, total_budget, K, C_init,tol,Randomize)
            result1 = MF.run_PDOO_fid1(POO_mult,Debug)
            isfid1 = False
            isExact = True
        if isfid1:
            MF = MFoptTree(mfobject, nu_max, rho_max, total_budget, K, C_init,tol,Randomize)
            result = MF.run_PDOO_fid1(POO_mult,Debug)
        else:
            if isExact:
                MF = MFoptTree(mfobject, nu_max, rho_max, total_budget, K, C_init,tol,Randomize)
                result = MF.run_PDOO_exact(POO_mult,Debug)
            else:
                MF = MFoptTree(mfobject, nu_max, rho_max, total_budget, K, C_init,tol,Randomize)
                result = MF.run_POO(POO_mult,Debug)
        R = R + [mfobject.opt_val - mfobject.eval_at_fidel_single_point_normalised([1.0], np.array(result[1]))]
        C = C + [result[2]/mfobject.eval_fidel_cost_single_point_normalised([1.0])]
        R1 = R1 + [mfobject.opt_val - mfobject.eval_at_fidel_single_point_normalised([1.0], np.array(result1[1]))]
        C1 = C1 + [result1[2]/mfobject.eval_fidel_cost_single_point_normalised([1.0])]

    if isPlot:
        plt.rc('text', usetex=True)
        plt.plot(C,R,'*-')
        if isAll:
            plt.plot(C1,R1,'o-')
            plt.legend(['FidelityTree','POO at $z = 1$'])
        plt.xlabel('Capital')
        plt.ylabel('Simple Regret')
        plt.grid()
        plt.savefig(image_name+'_actual.pdf')
        plt.close()

        plt.rc('text', usetex=True)
        plt.plot(C,np.log10(np.array(R)+1e-20),'*-')
        if isAll:
            plt.plot(C1,np.log10(np.array(R1)+1e-20),'o-')
            plt.legend(['FidelityTree','POO at $z = 1$'])
        plt.xlabel('Capital')
        plt.ylabel('Simple Regret (Log Scale)')
        plt.grid()
        plt.savefig(image_name+'_log10.pdf')
        plt.close()

    if isSave:
        dfdic = {}
        dfdic['cost_fid1'] = C1
        dfdic['cost'] = C
        dfdic['SR_fid1'] = R1
        dfdic['SR'] = R
        dfdic['SR_fid1_log10'] = list(np.log10(np.array(R1)+1e-20))
        dfdic['SR_log10'] = list(np.log10(np.array(R)+1e-20))
        df = pd.DataFrame(dfdic)
        df.to_csv(image_name + str('.csv'))

    return R,total_budget_mults
