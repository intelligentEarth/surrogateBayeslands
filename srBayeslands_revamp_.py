

#Main Contributer:   Rohitash Chandra  Email: c.rohitash@gmail.com 
# Other Contributers: Konark Jain, Arpit Kapoor, Ashray Aman

# Bayeslands II: Parallel tempering for multi-core systems - Badlands

from __future__ import print_function, division

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import random
import time
import operator
import math 
from pylab import rcParams
import copy
from copy import deepcopy 
from pylab import rcParams
import collections
from scipy import special
import fnmatch
import shutil
from PIL import Image
from io import StringIO
from cycler import cycler
import os
import shutil
import sys

import matplotlib.mlab as mlab
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import cKDTree
from scipy import stats 
#from pyBadlands.model import Model as badlandsModel


from badlands.model import Model as badlandsModel
import badlands 


from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import itertools
#import plotly
#import plotly.plotly as py
#from plotly.graph_objs import *
#plotly.offline.init_notebook_mode()
#from plotly.offline.offline import _plot_html
import pandas
import argparse

import pandas as pd
#import seaborn as sns


  
from scipy.ndimage import filters 

import scipy.ndimage as ndimage

from scipy.ndimage import gaussian_filter




from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.objectives import MSE, MAE
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from keras.models import load_model




#Initialise and parse inputs
parser=argparse.ArgumentParser(description='PTBayeslands modelling')

parser.add_argument('-p','--problem', help='Problem Number 1-crater-fast,2-crater,3-etopo-fast,4-etopo,5-null,6-mountain', required=True,   dest="problem",type=int)
parser.add_argument('-s','--samples', help='Number of samples', default=10000, dest="samples",type=int)
parser.add_argument('-r','--replicas', help='Number of chains/replicas, best to have one per availble core/cpu', default=10,dest="num_chains",type=int)
parser.add_argument('-t','--temperature', help='Demoninator to determine Max Temperature of chains (MT=no.chains*t) ', default=10,dest="mt_val",type=int)
parser.add_argument('-swap','--swap', help='Swap interval', dest="swap_interval",default= 2,type=int)
parser.add_argument('-b','--burn', help='How many samples to discard before determing posteriors', dest="burn_in",default=0.25,type=float)
parser.add_argument('-pt','--ptsamples', help='Ratio of PT vs straight MCMC samples to run', dest="pt_samples",default=0.5,type=float)  
parser.add_argument('-rain_intervals','--rain_intervals', help='rain_intervals', dest="rain_intervals",default=4,type=int)

parser.add_argument('-surrogate','--surrogate', help='Surrogate probability', dest="surrogate_prob",default=0.25,type=float)


parser.add_argument('-epsilon','--epsilon', help='epsilon for inital topo', dest="epsilon",default=0.5,type=float)



args = parser.parse_args()
    
#parameters for Parallel Tempering
problem = args.problem
samples = args.samples #10000  # total number of samples by all the chains (replicas) in parallel tempering
num_chains = args.num_chains
swap_interval = args.swap_interval
burn_in=args.burn_in
#maxtemp = int(num_chains * 5)/args.mt_val
maxtemp =   args.mt_val  
num_successive_topo = 4
pt_samples = args.pt_samples
epsilon = args.epsilon
rain_intervals = args.rain_intervals 
surrogate_prob = args.surrogate_prob

surrogate_int = int(epsilon * samples/num_chains)  # surrogate interval

print(surrogate_int, ' is surrogate interval')

print(surrogate_prob, ' is surrogate prob') 

 
class surrogate: #General Class for surrogate models for predicting likelihood given the weights

    def __init__(self, model, X, Y, min_X, max_X, min_Y , max_Y, path, save_surrogate_data, model_topology):

        self.path = path + '/surrogate'
        indices = np.where(Y==np.inf)[0]
        X = np.delete(X, indices, axis=0)
        Y = np.delete(Y, indices, axis=0)
        self.model_signature = 0.0
        self.X = X
        self.Y = Y
        self.min_Y = min_Y
        self.max_Y = max_Y
        self.min_X = min_X
        self.max_X = max_X

        self.model_topology = model_topology

        self.save_surrogate_data =  save_surrogate_data

        if model=="gp":
            self.model_id = 1
        elif model == "nn":
            self.model_id = 2
        elif model == "krnn": # keras nn
            self.model_id = 3
            self.krnn = Sequential()
        else:
            print("Invalid Model!")

    def normalize(self, X):
        maxer = np.zeros((1,X.shape[1]))
        miner = np.ones((1,X.shape[1]))

        for i in range(X.shape[1]):
            maxer[0,i] = max(X[:,i])
            miner[0,i] = min(X[:,i])
            X[:,i] = (X[:,i] - min(X[:,i]))/(max(X[:,i]) - min(X[:,i]))
        return X, maxer, miner

    def create_model(self):
        krnn = Sequential()

        if self.model_topology == 1:
            krnn.add(Dense(64, input_dim=self.X.shape[1], kernel_initializer='uniform', activation ='relu')) #64
            krnn.add(Dense(16, kernel_initializer='uniform', activation='relu'))  #16

        if self.model_topology == 2:
            krnn.add(Dense(120, input_dim=self.X.shape[1], kernel_initializer='uniform', activation ='relu')) #64
            krnn.add(Dense(40, kernel_initializer='uniform', activation='relu'))  #16

        if self.model_topology == 3:
            krnn.add(Dense(200, input_dim=self.X.shape[1], kernel_initializer='uniform', activation ='relu')) #64
            krnn.add(Dense(50, kernel_initializer='uniform', activation='relu'))  #16

        krnn.add(Dense(1, kernel_initializer ='uniform', activation='sigmoid'))
        return krnn

    def train(self, model_signature):
        #X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.10, random_state=42)

        X_train = self.X
        X_test = self.X
        y_train = self.Y
        y_test =  self.Y #train_test_split(self.X, self.Y, test_size=0.10, random_state=42)

        self.model_signature = model_signature


        if self.model_id is 3:
            if self.model_signature==1.0:
                self.krnn = self.create_model()
            else:
                while True:
                    try:
                        # You can see two options to initialize model now. If you uncomment the first line then the model id loaded at every time with stored weights. On the other hand if you uncomment the second line a new model will be created every time without the knowledge from previous training. This is basically the third scheme we talked about for surrogate experiments.
                        # To implement the second scheme you need to combine the data from each training.

                        self.krnn = load_model(self.path+'/model_krnn_%s_.h5'%(model_signature-1))
                        #self.krnn = self.create_model()
                        break
                    except EnvironmentError as e:
                        # pass
                        # # print(e.errno)
                        # time.sleep(1)
                        print ('ERROR in loading latest surrogate model, loading previous one in TRAIN')

            early_stopping = EarlyStopping(monitor='val_loss', patience=5)
            self.krnn.compile(loss='mse', optimizer='adam', metrics=['mse'])
            train_log = self.krnn.fit(X_train, y_train.ravel(), batch_size=50, epochs=20, validation_split=0.1, verbose=0, callbacks=[early_stopping])

            scores = self.krnn.evaluate(X_test, y_test.ravel(), verbose = 0)
            # print("%s: %.5f" % (self.krnn.metrics_names[1], scores[1]))

            self.krnn.save(self.path+'/model_krnn_%s_.h5' %self.model_signature)
            # print("Saved model to disk  ", self.model_signature)
 

            results = np.array([scores[1]])
            # print(results, 'train-metrics')


            with open(('%s/train_metrics.txt' % (self.path)),'ab') as outfile:
                np.savetxt(outfile, results)

            if self.save_surrogate_data is True:
                with open(('%s/learnsurrogate_data/X_train.csv' % (self.path)),'ab') as outfile:
                    np.savetxt(outfile, X_train)
                with open(('%s/learnsurrogate_data/Y_train.csv' % (self.path)),'ab') as outfile:
                    np.savetxt(outfile, y_train)
                with open(('%s/learnsurrogate_data/X_test.csv' % (self.path)),'ab') as outfile:
                    np.savetxt(outfile, X_test)
                with open(('%s/learnsurrogate_data/Y_test.csv' % (self.path)),'ab') as outfile:
                    np.savetxt(outfile, y_test)

    def predict(self, X_load, initialized):


        if self.model_id == 3:

            if initialized == False:
                model_sign = np.loadtxt(self.path+'/model_signature.txt')
                self.model_signature = model_sign
                while True:
                    try:
                        self.krnn = load_model(self.path+'/model_krnn_%s_.h5'%self.model_signature)
                        # # print (' Tried to load file : ', self.path+'/model_krnn_%s_.h5'%self.model_signature)
                        break
                    except EnvironmentError as e:
                        print(e)
                        # pass

                self.krnn.compile(loss='mse', optimizer='rmsprop', metrics=['mse'])
                krnn_prediction =-1.0
                prediction = -1.0

            else:
                krnn_prediction = self.krnn.predict(X_load)[0]
                prediction = krnn_prediction*(self.max_Y[0,0]-self.min_Y[0,0]) + self.min_Y[0,0]

            return prediction, krnn_prediction



class ptReplica(multiprocessing.Process):
    def __init__(self,   num_param, vec_parameters,    minlimits_vec, maxlimits_vec,
     stepratio_vec,   check_likelihood_sed ,  swap_interval, sim_interval, simtime, samples, real_elev,  real_erodep_pts, erodep_coords, filename, 
     xmlinput,  run_nb, tempr, parameter_queue,event , main_proc,   burn_in, surrogate_parameterqueue, surrogate_interval,surrogate_prob, surrogate_start,
        surrogate_resume, save_surrogatedata, use_surrogate, compare_surrogate, pause_chain_event, resume_chain_event, surrogate_topology):

        multiprocessing.Process.__init__(self)
        self.processID = tempr      
        self.parameter_queue = parameter_queue
        self.event = event
        self.signal_main = main_proc
        self.temperature = tempr
        self.swap_interval = swap_interval
        self.folder = filename
        self.input = xmlinput  
        self.simtime = simtime
        self.samples = samples
        self.run_nb = run_nb 
        self.num_param =  num_param
        self.font = 9
        self.width = 1 
        self.vec_parameters = np.asarray(vec_parameters)
        self.minlimits_vec = np.asarray(minlimits_vec)
        self.maxlimits_vec = np.asarray(maxlimits_vec)
        self.stepratio_vec = np.asarray(stepratio_vec)
        self.check_likelihood_sed =  check_likelihood_sed
        self.real_erodep_pts = real_erodep_pts
        self.erodep_coords = erodep_coords
        self.real_elev = real_elev
        self.runninghisto = True  
        self.burn_in = burn_in
        self.sim_interval = sim_interval
        self.sedscalingfactor = 50 # this is to ensure that the sediment likelihood is given more emphasis as it considers fewer points (dozens of points) when compared to elev liklihood (thousands of points)
        self.adapttemp =  self.temperature


        
        self.minlim_param = minlimits_vec
        self.maxlim_param = maxlimits_vec


        self.surrogate_topology = surrogate_topology



 

        self.minY = np.zeros((1,1))
        self.maxY = np.ones((1,1)) 

        self.compare_surrogate = True

                #SURROGATE VARIABLES
        self.surrogate_parameter_queue = surrogate_parameterqueue
        self.surrogate_start = surrogate_start
        self.surrogate_resume = surrogate_resume
        self.surrogate_interval = surrogate_interval
        self.surrogate_prob = surrogate_prob
        self.save_surrogate_data = save_surrogatedata
        self.use_surrogate = use_surrogate


        self.pause_chain_event = pause_chain_event

        self.resume_chain_event = resume_chain_event


        self.stepsize_vec = np.zeros(self.maxlimits_vec.size)



        self.adapt_cov = 40  # try make it around 50 (frequency of updating adap mat)
        self.cholesky = [] 
        self.cov_init = False  # dont change, keep it as false
        self.use_cov = True # Make True if you wish to use Adaptive RW proposals (Better for convergence as shown in paper)
        self.cov_counter = 0 


    def interpolateArray(self, coords=None, z=None, dz=None):
        """
        Interpolate the irregular spaced dataset from badlands on a regular grid.
        """
        x, y = np.hsplit(coords, 2)
        dx = (x[1]-x[0])[0]

        nx = int((x.max() - x.min())/dx+1)
        ny = int((y.max() - y.min())/dx+1)
        xi = np.linspace(x.min(), x.max(), nx)
        yi = np.linspace(y.min(), y.max(), ny)

        xi, yi = np.meshgrid(xi, yi)
        xyi = np.dstack([xi.flatten(), yi.flatten()])[0]
        XY = np.column_stack((x,y))

        tree = cKDTree(XY)
        distances, indices = tree.query(xyi, k=3)
        if len(z[indices].shape) == 3:
            z_vals = z[indices][:,:,0]
            dz_vals = dz[indices][:,:,0]
        else:
            z_vals = z[indices]
            dz_vals = dz[indices]

        zi = np.average(z_vals,weights=(1./distances), axis=1)
        dzi = np.average(dz_vals,weights=(1./distances), axis=1)
        onIDs = np.where(distances[:,0] == 0)[0]
        if len(onIDs) > 0:
            zi[onIDs] = z[indices[onIDs,0]]
            dzi[onIDs] = dz[indices[onIDs,0]]
        zreg = np.reshape(zi,(ny,nx))
        dzreg = np.reshape(dzi,(ny,nx))
        return zreg,dzreg





    def run_badlands(self, input_vector):

        model = badlandsModel()

        # Load the XmL input file
        model.load_xml(str(self.run_nb), self.input, muted=True)

        # Adjust erodibility based on given parameter
        model.input.SPLero = input_vector[1] 
        model.flow.erodibility.fill(input_vector[1])
 
        model.force.rainVal[:] = input_vector[0] 
 
        model.input.SPLm = input_vector[2] 
        model.input.SPLn = input_vector[3] 

 
 
        if self.num_param == 5: # Mountain
            #Round the input vector 
            #k=round(input_vector[4]*2)/2 #to closest 0.5  

            k=round(input_vector[4],1) #to closest 0.1


            #Load the current tectonic uplift parameters
            tectonicValues=pandas.read_csv(str(model.input.tectFile[0]),sep=r'\s+',header=None,dtype=np.float).values
        
            #Adjust the parameters by our value k, and save them out
            newFile = "Examples/mountain_nath/mountaindata/tect/uplift"+str(self.temperature)+"_"+str(k)+".csv"
            newtect = pandas.DataFrame(tectonicValues*k)
            newtect.to_csv(newFile,index=False,header=False)
            
            #Update the model uplift tectonic values
            model.input.tectFile[0]=newFile
            #print(model.input.tectFile)
        else: 

            model.input.CDm = input_vector[4] # submarine diffusion
            model.input.CDa = input_vector[5] # aerial diffusion




        elev_vec = collections.OrderedDict()
        erodep_vec = collections.OrderedDict()
        erodep_pts_vec = collections.OrderedDict()

        for x in range(len(self.sim_interval)):
            self.simtime = self.sim_interval[x]

            model.run_to_time(self.simtime, muted=True)
         
            #elev, erodep = self.interpolate_array(model.FVmesh.node_coords[:, :2], model.elevation, model.cumdiff)


            elev, erodep = self.interpolateArray(model.FVmesh.node_coords[:, :2], model.elevation, model.cumdiff)

            erodep_pts = np.zeros((self.erodep_coords.shape[0]))

            for count, val in enumerate(self.erodep_coords):
                erodep_pts[count] = erodep[val[0], val[1]]

            elev_vec[self.simtime] = elev
            erodep_vec[self.simtime] = erodep
            erodep_pts_vec[self.simtime] = erodep_pts

        return elev_vec, erodep_vec, erodep_pts_vec
 

 

 
    def likelihood_func(self,input_vector ):
        #print("Running likelihood function: ", input_vector)
        
        pred_elev_vec, pred_erodep_vec, pred_erodep_pts_vec = self.run_badlands(input_vector )
        tausq = np.sum(np.square(pred_elev_vec[self.simtime] - self.real_elev))/self.real_elev.size 
        tau_erodep =  np.zeros(self.sim_interval.size) 
        #print(self.sim_interval.size, self.real_erodep_pts.shape)
        for i in range(  self.sim_interval.size):
            tau_erodep[i]  =  np.sum(np.square(pred_erodep_pts_vec[self.sim_interval[i]] - self.real_erodep_pts[i]))/ self.real_erodep_pts.shape[1]

        likelihood_elev = - 0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(pred_elev_vec[self.simtime] - self.real_elev) / tausq 
        likelihood_erodep = 0 
        
        if self.check_likelihood_sed  == True: 

            for i in range(1, self.sim_interval.size):
                likelihood_erodep  += np.sum(-0.5 * np.log(2 * math.pi * tau_erodep[i]) - 0.5 * np.square(pred_erodep_pts_vec[self.sim_interval[i]] - self.real_erodep_pts[i]) / tau_erodep[i]) # only considers point or core of erodep
        
            likelihood = np.sum(likelihood_elev) +  (likelihood_erodep * self.sedscalingfactor)

        else:
            likelihood = np.sum(likelihood_elev)


        #print(pred_erodep_pts_vec,  ' pred_erodep_pts_vec')

        rmse_elev = np.sqrt(tausq)
        rmse_erodep = np.sqrt(tau_erodep) 
        avg_rmse_er = np.average(rmse_erodep)

 

        return [likelihood *(1.0/self.adapttemp), pred_elev_vec, pred_erodep_pts_vec, likelihood, rmse_elev, avg_rmse_er]


    def computeCovariance(self, i, pos_v):
        cov_mat = np.cov(pos_v[:i,].T)
        # np.savetxt('%s/cov_mat_%s.txt' %(self.filename,self.temperature), cov_mat )
        # print ('\n step ratio vec', self.stepratio_vec)
        #print ('step size vec', self.stepsize_vec, '\n')

        cov_noise_old = (self.stepratio_vec * self.stepratio_vec)*np.identity(cov_mat.shape[0], dtype = float)
        cov_noise = self.stepsize_vec*np.identity(cov_mat.shape[0], dtype = float)
        
        #print ('\ncov_noise_old', cov_noise_old)
        #print ('cov_noise_new', cov_noise, '\n')

        covariance = np.add(cov_mat, cov_noise)    

        #print(covariance, ' covariance')    
        L = np.linalg.cholesky(covariance)
        self.cholesky = L
        self.cov_init = True
        # self.cov_counter += 1 




    def run(self):

        #This is a chain that is distributed to many cores. AKA a 'Replica' in Parallel Tempering

        samples = self.samples
        count_list = [] 
        stepsize_vec = np.zeros(self.maxlimits_vec.size)
        span = (self.maxlimits_vec-self.minlimits_vec) 



        for i in range(stepsize_vec.size): # calculate the step size of each of the parameters
            stepsize_vec[i] = self.stepratio_vec[i] * span[i]

        self.stepsize_vec = stepsize_vec




        v_proposal = self.vec_parameters # initial param values passed to badlands
        v_current = v_proposal # to give initial value of the chain

        #  initial predictions from Badlands model
        #print("Intital parameter predictions: ", v_current)
        ##initial_predicted_elev, initial_predicted_erodep, init_pred_erodep_pts_vec = self.run_badlands(v_current)
        
        #calc initial likelihood with initial parameters
        [likelihood, predicted_elev,  pred_erodep_pts, likl_without_temp, avg_rmse_el, avg_rmse_er] = self.likelihood_func(v_current )

        print('\tinitial likelihood:', likelihood)

        likeh_list = np.zeros((samples,2)) # one for posterior of likelihood and the other for all proposed likelihood
        likeh_list[0,:] = [-10000, -10000] # to avoid prob in calc of 5th and 95th percentile   later

        count_list.append(0) # just to count number of accepted for each chain (replica)
        accept_list = np.zeros(samples)
        

        #---------------------------------------
        #now, create memory to save all the accepted tau proposals
        prev_accepted_elev = deepcopy(predicted_elev)
        prev_acpt_erodep_pts = deepcopy(pred_erodep_pts) 
        sum_elev = deepcopy(predicted_elev)
        sum_erodep_pts = deepcopy(pred_erodep_pts)

        #print('time to change')
        burnsamples = int(samples*self.burn_in)
        
        #---------------------------------------
        #now, create memory to save all the accepted   proposals of rain, erod, etc etc, plus likelihood
        pos_param = np.zeros((samples,v_current.size)) 
        list_yslicepred = np.zeros((samples,self.real_elev.shape[0]))  # slice mid y axis  
        list_xslicepred = np.zeros((samples,self.real_elev.shape[1])) # slice mid x axis  
        ymid = int(self.real_elev.shape[1]/2 ) 
        xmid = int(self.real_elev.shape[0]/2)
        list_erodep  = np.zeros((samples,pred_erodep_pts[self.simtime].size))
        list_erodep_time  = np.zeros((samples , self.sim_interval.size , pred_erodep_pts[self.simtime].size))

        start = time.time() 

        num_accepted = 0
        num_div = 0 

        #pt_samples = samples * 0.5 # this means that PT in canonical form with adaptive temp will work till pt  samples are reached. Set in arguments, default 0.5

        init_count = 0

        rmse_elev  = np.zeros(samples)
        rmse_erodep = np.zeros(samples)


        s_pos_w = np.ones((samples, v_proposal.size)) #Surrogate Trainer
        lhood_list = np.zeros((samples,1))
        #surrogate_list = np.zeros((samples ,1))
 
        is_true_lhood = True


        lhood_counter = 0
        lhood_counter_inf = 0
        reject_counter = 0
        reject_counter_inf = 0
 

        pt_samples = samples * 1# this means that PT in canonical form with adaptive temp will work till pt  samples are reached

  
        trainset_empty = True 

        surrogate_model = None 
        surrogate_counter = 0 
        naccept  = 0

        likeh_list = np.zeros((samples,2))      # Index 0 -> For posterior samples likelihood // Index 1 -> All proposed likelihood
        likeh_list[0,:] = [-100, -100]          # Initialised in order to calc 5th and 95th percentile later
        surg_likeh_list = np.zeros((samples,3)) # Index 0 -> All fwd model Likl// Index 1 ->Surrogate Likelihood values


        '''Parameter Storage'''
        prop_list = np.zeros((samples,v_current.size))      # Proposed params
        pos_param = np.zeros((samples,v_current.size))      # Accepted proposal params


        local_model_signature = 0.0

 


        #if trainset_empty == True:
        surr_train_set = np.zeros((1000, self.num_param+1))


        self.resume_chain_event.clear()

        count_real = 0


 
 

        for i in range(samples-1):

            print ("Temperature: ", self.temperature, ' Sample: ', i ,"/",samples)

            if i < pt_samples:
                self.adapttemp =  self.temperature #* ratio  #

            if i == pt_samples and init_count ==0: # move to MCMC canonical
                self.adapttemp = 1
                [likelihood, predicted_elev,  pred_erodep_pts, likl_without_temp, avg_rmse_el, avg_rmse_er] = self.likelihood_func(v_proposal) 
                init_count = 1


            if self.cov_init and self.use_cov:        
                v_p = np.random.normal(size = v_current.shape)
                v_proposal = v_current + np.dot(self.cholesky,v_p)   # Adaptive RW proposals 
            else: 
                v_proposal =  np.random.normal(v_current,stepsize_vec)  # RW proposals


            for j in range(v_current.size):
                if v_proposal[j] > self.maxlimits_vec[j]:
                    v_proposal[j] = v_current[j]
                elif v_proposal[j] < self.minlimits_vec[j]:
                    v_proposal[j] = v_current[j]

            ku = random.uniform(0,1)

            surrogate_X = v_proposal
            surrogate_Y = np.array([likelihood])


            if ku<self.surrogate_prob and i>=self.surrogate_interval+1:

                is_true_lhood = False

                if surrogate_model == None:
                    minmax = np.loadtxt(self.folder+'/surrogate/minmax.txt')
                    self.minY[0,0] = minmax[0]
                    self.maxY[0,0] = minmax[1]
                    surrogate_model = surrogate("krnn",surrogate_X.copy(),surrogate_Y.copy(), self.minlim_param, self.maxlim_param, self.minY, self.maxY, self.folder, self.save_surrogate_data,self.surrogate_topology  )
                    surrogate_likelihood, nn_predict = surrogate_model.predict(v_proposal.reshape(1,v_proposal.shape[0]),False)
                    surrogate_likelihood = surrogate_likelihood *(1.0/self.adapttemp)

                    


                elif self.surrogate_init == 0.0:
                    surrogate_likelihood,  nn_predict = surrogate_model.predict(v_proposal.reshape(1,v_proposal.shape[0]), False )
                    surrogate_likelihood = surrogate_likelihood *(1.0/self.adapttemp)
                else:
                    surrogate_likelihood,  nn_predict = surrogate_model.predict(v_proposal.reshape(1,v_proposal.shape[0]), True )
                    surrogate_likelihood = surrogate_likelihood *(1.0/self.adapttemp)



                likelihood_mov_ave = (surg_likeh_list[i,2] + surg_likeh_list[i-1,2]+ surg_likeh_list[i-2,2])/3
                likelihood_proposal = (surrogate_likelihood  * 0.5 ) + (  likelihood_mov_ave * 0.5)
                #print(likelihood_proposal, likelihood_mov_ave, surrogate_likelihood[0],  self.temperature, 'likelihood_proposal, likelihood_mov_ave, surrogate_likelihood[0] : is likelihood proposal')


                if self.compare_surrogate is True: 
                    [likelihood_proposal_true, predicted_elev, pred_erodep_pts, likl_without_temp, avg_rmse_el, avg_rmse_er] = self.likelihood_func(v_proposal) 
                else:
                    likelihood_proposal_true = 0


                print ('\nSample : ', i, ' Chain :', self.adapttemp, ' -A', likelihood_proposal_true, ' vs. P ',  likelihood_proposal, ' ---- nnPred ', nn_predict, self.minY, self.maxY )
                
                surrogate_counter += 1
                surg_likeh_list[i+1,0] =  likelihood_proposal_true
                surg_likeh_list[i+1,1] = likelihood_proposal
                surg_likeh_list[i+1,2] = likelihood_mov_ave
   


            else:
                is_true_lhood = True
                trainset_empty = False

                surg_likeh_list[i+1,1] =  np.nan
 
                [likelihood_proposal, predicted_elev, pred_erodep_pts, likl_without_temp, avg_rmse_el, avg_rmse_er] = self.likelihood_func(v_proposal)

                likl_wo_temp = np.array([likl_without_temp])
                X, Y = v_proposal,likl_wo_temp
                X = X.reshape(1, X.shape[0])
                Y = Y.reshape(1, Y.shape[0])
 
                param_train = np.concatenate([X, Y],axis=1)
                #surr_train_set = np.vstack((surr_train_set, param_train))

                surr_train_set[count_real, :] = param_train
                count_real = count_real +1
 





                surg_likeh_list[i+1,0] = likelihood_proposal
                surg_likeh_list[i+1,2] = likelihood_proposal 




            #[likelihood_proposal, predicted_elev,  pred_erodep_pts, likl_without_temp, avg_rmse_el, avg_rmse_er] = self.likelihood_func(v_proposal)

            final_predtopo= predicted_elev[self.simtime]
            pred_erodep = pred_erodep_pts[self.simtime]

            # Difference in likelihood from previous accepted proposal
            diff_likelihood = likelihood_proposal - likelihood

            try:
                mh_prob = min(1, math.exp(diff_likelihood))
            except OverflowError as e:
                mh_prob = 1

            u = random.uniform(0,1)
            
            accept_list[i+1] = num_accepted
            likeh_list[i+1,0] = likelihood_proposal

            prop_list[i+1,] = v_proposal

            if u < mh_prob: # Accept sample
                # Append sample number to accepted list
                count_list.append(i)            
                
                likelihood = likelihood_proposal
                v_current = v_proposal
                pos_param[i+1,:] = v_current # features rain, erodibility and others  (random walks is only done for this vector)
                likeh_list[i + 1,1]=likelihood  # contains  all proposal liklihood (accepted and rejected ones)
                list_yslicepred[i+1,:] =  final_predtopo[:, ymid] # slice taken at mid of topography along y axis  
                list_xslicepred[i+1,:]=   final_predtopo[xmid, :]  # slice taken at mid of topography along x axis 
                #list_erodep[i+1,:] = pred_erodep
                rmse_elev[i+1,] = avg_rmse_el
                rmse_erodep[i+1,] = avg_rmse_er

                print(self.temperature, i, likelihood , avg_rmse_el, avg_rmse_er, '   --------- ')

                for x in range(self.sim_interval.size): 
                    list_erodep_time[i+1,x, :] = pred_erodep_pts[self.sim_interval[x]]

                num_accepted = num_accepted + 1 
                prev_accepted_elev.update(predicted_elev)

                if i>burnsamples: 
                    
                    for k, v in prev_accepted_elev.items():
                        sum_elev[k] += v 

                    for k, v in pred_erodep_pts.items():
                        sum_erodep_pts[k] += v

                    num_div += 1

            else: # Reject sample
                likeh_list[i + 1, 1]=likeh_list[i,1] 
                pos_param[i+1,:] = pos_param[i,:]
                list_yslicepred[i+1,:] =  list_yslicepred[i,:] 
                list_xslicepred[i+1,:]=   list_xslicepred[i,:]
                list_erodep[i+1,:] = list_erodep[i,:]
                list_erodep_time[i+1,:, :] = list_erodep_time[i,:, :]
                rmse_elev[i+1,] = rmse_elev[i,] 
                rmse_erodep[i+1,] = rmse_erodep[i,]

            
                if i>burnsamples:

                    for k, v in prev_accepted_elev.items():
                        sum_elev[k] += v

                    for k, v in prev_acpt_erodep_pts.items():
                        sum_erodep_pts[k] += v

                    num_div += 1
 

 
            eta = 1


            if (i >= self.adapt_cov and i % self.adapt_cov == 0) :
                print ('\ncov computed = i ',i, '\n')
                self.computeCovariance(i,pos_param)

 



            if i%self.surrogate_interval == 0 and i != 0:
                print("\n\nSample:{}\n\n".format(i))
                #param = np.concatenate([v_current, np.asarray([eta]).reshape(1), np.asarray([likelihood*self.adapttemp]),np.asarray([self.adapttemp]),np.asarray([i])])
                # add parameters to the swap param queue and surrogate params queue
                #self.parameter_queue.put(param)

                surr_train = surr_train_set[1:count_real, :]
 

                #self.surrogate_parameter_queue.put(all_param)

                self.surrogate_parameter_queue.put(surr_train)
                # Pause the chain execution and signal main process
                self.pause_chain_event.set()
                print("Temperature: {} waiting for swap and surrogate training complete signal. Event: {}".format(self.temperature, self.pause_chain_event.is_set()))
                # Wait for the main process to complete the swap and surrogate training
                self.resume_chain_event.clear()
                self.resume_chain_event.wait()
                # retrieve parameters fom queues if it has been swapped
                ''' comment below 2 lines to stop swap '''
                #result =  self.parameter_queue.get()
                #v_current= result[0:v_current.size]
                
                #eta = result[w.size]
                #likelihood = result[w.size+1]/self.adapttemp

                model_sign = np.loadtxt(self.folder+'/surrogate/model_signature.txt')
                self.model_signature = model_sign
                #print("model_signature updated")

                if self.model_signature==1.0:
                    minmax = np.loadtxt(self.folder+'/surrogate/minmax.txt')
                    self.minY[0,0] = minmax[0]
                    self.maxY[0,0] = minmax[1]
                    # # print 'min ', self.minY, ' max ', self.maxY
                    dummy_X = np.zeros((1,1))
                    dummy_Y = np.zeros((1,1))
                    surrogate_model = surrogate("krnn", dummy_X, dummy_Y, self.minlim_param, self.maxlim_param, self.minY, self.maxY, self.folder, self.save_surrogate_data, self.surrogate_topology )

                self.surrogate_init,  nn_predict  = surrogate_model.predict(v_proposal.reshape(1,v_proposal.shape[0]), False) 

                #del surr_train_set
                trainset_empty = True 



                np.savetxt(self.folder+'/surrogate/traindata_'+ str(int(self.temperature*10)) +'_'+str(local_model_signature)    +'_.txt', surr_train_set)


                #surr_train_set = np.zeros((1, self.num_param+1))

                count_real = 0


            #parameters= np.concatenate([v_current, np.asarray([eta]).reshape(1), np.asarray([likelihood]), np.asarray([self.adapttemp]), np.asarray([i])])
            #self.parameter_queue.put(parameters)
             

            save_res =  np.array([i, num_accepted, likelihood, likelihood_proposal, rmse_elev[i+1,], rmse_erodep[i+1,]])  
 


            outfilex = open(('%s/posterior/pos_parameters/stream_chain_%s.txt' % (self.folder, self.temperature)), "a") 
            x = np.array([pos_param[i+1,:]]) 
            np.savetxt(outfilex,x, fmt='%1.8f')  

            outfile1 = open(('%s/posterior/predicted_topo/x_slice/stream_xslice_%s.txt' % (self.folder, self.temperature)), "a") 
            np.savetxt(outfile1,np.array([list_xslicepred[i+1,:]]), fmt='%1.2f')  

            outfile2 = open(('%s/posterior/predicted_topo/y_slice/stream_yslice_%s.txt' % (self.folder, self.temperature)), "a") 
            np.savetxt(outfile2,np.array([list_yslicepred[i+1,:]]), fmt='%1.2f') 
 
            outfile3 = open(('%s/posterior/stream_res_%s.txt' % (self.folder, self.temperature)), "a") 
            np.savetxt(outfile3,np.array([save_res]), fmt='%1.2f')  
 
            outfile4 = open( ('%s/performance/lhood/stream_res_%s.txt' % (self.folder, self.temperature)), "a") 
            np.savetxt(outfile4,np.array([likeh_list[i + 1,0]]), fmt='%1.2f') 

            outfile5 = open( ('%s/performance/accept/stream_res_%s.txt' % (self.folder, self.temperature)), "a")  
            np.savetxt(outfile5,np.array([accept_list[i+1]]), fmt='%1.2f')

            outfile6 = open(  ('%s/performance/rmse_edep/stream_res_%s.txt' % (self.folder, self.temperature)), "a")  
            np.savetxt(outfile6,np.array([rmse_erodep[i+1,]]), fmt='%1.2f')

            outfile7 = open( ('%s/performance/rmse_elev/stream_res_%s.txt' % (self.folder, self.temperature)), "a")  
            np.savetxt(outfile7,np.array([rmse_elev[i+1,]]), fmt='%1.2f')


            outfile8 = open( ( '%s/posterior/surg_likelihood/stream_res_%s.txt' % (self.folder, self.temperature)), "a") 
            #with file(('%s/posterior/surg_likelihood/stream_res_%s.txt' % (self.folder, self.temperature)),'a') as outfile:
            np.savetxt(outfile8,np.array([surg_likeh_list[i+1,]]), fmt='%1.2f')

            #file_name = self.folder+'/posterior/surg_likelihood/chain_'+ str(self.temperature)+ '.txt'
            #np.savetxt(file_name,surg_likeh_list, fmt='%1.4f')
 

            temp = list_erodep_time[i+1,:,:]  
            #print(temp, 'before')
            temp = temp.flatten() # np.reshape(temp, temp.shape[0]*1) 
            #print(temp, 'after')
 
            outfile10 = open( (self.folder + '/posterior/predicted_topo/sed/chain_' + str(self.temperature) + '.txt'), "a") 
            np.savetxt(outfile10, np.array([temp]), fmt='%1.2f') 

 
            #file_name = self.folder + '/posterior/predicted_topo/sed/chain_' + str(self.temperature) + '.txt'
            '''with file(file_name ,'a') as outfile:
                np.savetxt(outfile, np.array([temp]), fmt='%1.2f') '''
 


        accepted_count =  len(count_list) 
        accept_ratio = accepted_count / (samples * 1.0) * 100
        others = np.asarray([ likelihood])
        param = np.concatenate([v_current,others,np.asarray([self.temperature])])   

        '''print("param first:",param)
        print("v_current",v_current)
        print("others",others)
        print("temp",np.asarray([self.temperature]))'''
        
        self.parameter_queue.put(param)

         

        for k, v in sum_elev.items():
            sum_elev[k] = np.divide(sum_elev[k], num_div)
            mean_pred_elevation = sum_elev[k]

            sum_erodep_pts[k] = np.divide(sum_erodep_pts[k], num_div)
            mean_pred_erodep_pnts = sum_erodep_pts[k]

            file_name = self.folder + '/posterior/predicted_topo/topo/chain_' + str(k) + '_' + str(self.temperature) + '.txt'
            np.savetxt(file_name, mean_pred_elevation, fmt='%.2f')

        self.signal_main.set()


class ParallelTempering:

    def __init__(self,  vec_parameters,  minlimits_vec, maxlimits_vec, stepratio_vec,    num_chains, maxtemp,NumSample,swap_interval, fname, 
        realvalues_vec, num_param,  real_elev, erodep_pts, erodep_coords, simtime, siminterval, resolu_factor, run_nb, inputxml, surrogate_interval, surrogate_prob,
         save_surrogatedata, use_surrogate, compare_surrogate ):


        self.swap_interval = swap_interval
        self.folder = fname
        self.maxtemp = maxtemp
        self.num_swap = 0
        self.num_chains = num_chains
        self.chains = []

        self.surrogate_chains = []

        self.temperatures = []
        self.NumSamples = int(NumSample/self.num_chains)
        self.sub_sample_size = max(1, int( 0.05* self.NumSamples))
        self.show_fulluncertainity = False # needed in cases when you reall want to see full prediction of 5th and 95th percentile of topo. takes more space 
        self.real_erodep_pts  = erodep_pts
        self.real_elev = real_elev
        self.resolu_factor =  resolu_factor
        self.num_param = num_param
        self.erodep_coords = erodep_coords
        self.simtime = simtime
        self.sim_interval = siminterval
        self.run_nb =run_nb 
        self.xmlinput = inputxml
        self.vec_parameters = vec_parameters
        self.realvalues  =  realvalues_vec 

        self.minlimits_vec = minlimits_vec 
        self.maxlimits_vec = maxlimits_vec
        self.stepratio_vec = stepratio_vec


        self.surrogate_topology = 1



        
        # create queues for transfer of parameters between process chain
        #self.chain_parameters = [multiprocessing.Queue() for i in range(0, self.num_chains) ]
        self.parameter_queue = [multiprocessing.Queue() for i in range(num_chains)]
        self.chain_queue = multiprocessing.JoinableQueue()  
        self.wait_chain = [multiprocessing.Event() for i in range (self.num_chains)]

        # two ways events are used to synchronize chains
        self.event = [multiprocessing.Event() for i in range (self.num_chains)]
        #self.wait_chain = [multiprocessing.Event() for i in range (self.num_chains)]



        self.pause_chain_events = [multiprocessing.Event() for i in range (self.num_chains)]
        self.resume_chain_events = [multiprocessing.Event() for i in range (self.num_chains)]



        self.surrogate_interval = surrogate_interval
        self.surrogate_prob = surrogate_prob
        self.surrogate_resume_events = [multiprocessing.Event() for i in range(self.num_chains)]
        self.surrogate_start_events = [multiprocessing.Event() for i in range(self.num_chains)]
        self.surrogate_parameter_queues = [multiprocessing.Queue() for i in range(self.num_chains)]
        self.surrchain_queue = multiprocessing.JoinableQueue()

        self.minY = np.zeros((1,1))
        self.maxY = np.ones((1,1))

        self.model_signature = 0.0
        self.save_surrogate_data =  save_surrogatedata
        self.use_surrogate = use_surrogate 
        self.compare_surrogate = compare_surrogate


        #surrogate_interval, surrogate_prob, save_surrogatedata, use_surrogate, compare_surrogate



        self.geometric =  True
        self.total_swap_proposals = 0

         


    def default_beta_ladder(self, ndim, ntemps, Tmax): #https://github.com/konqr/ptemcee/blob/master/ptemcee/sampler.py
        """
        Returns a ladder of :math:`\beta \equiv 1/T` under a geometric spacing that is determined by the
        arguments ``ntemps`` and ``Tmax``.  The temperature selection algorithm works as follows:
        Ideally, ``Tmax`` should be specified such that the tempered posterior looks like the prior at
        this temperature.  If using adaptive parallel tempering, per `arXiv:1501.05823
        <http://arxiv.org/abs/1501.05823>`_, choosing ``Tmax = inf`` is a safe bet, so long as
        ``ntemps`` is also specified.
        
        """

        if type(ndim) != int or ndim < 1:
            raise ValueError('Invalid number of dimensions specified.')
        if ntemps is None and Tmax is None:
            raise ValueError('Must specify one of ``ntemps`` and ``Tmax``.')
        if Tmax is not None and Tmax <= 1:
            raise ValueError('``Tmax`` must be greater than 1.')
        if ntemps is not None and (type(ntemps) != int or ntemps < 1):
            raise ValueError('Invalid number of temperatures specified.')

        tstep = np.array([25.2741, 7., 4.47502, 3.5236, 3.0232,
                        2.71225, 2.49879, 2.34226, 2.22198, 2.12628,
                        2.04807, 1.98276, 1.92728, 1.87946, 1.83774,
                        1.80096, 1.76826, 1.73895, 1.7125, 1.68849,
                        1.66657, 1.64647, 1.62795, 1.61083, 1.59494 ])

        if ndim > tstep.shape[0]:
            # An approximation to the temperature step at large
            # dimension
            tstep = 1.0 + 2.0*np.sqrt(np.log(4.0))/np.sqrt(ndim)
        else:
            tstep = tstep[ndim-1]

        appendInf = False
        if Tmax == np.inf:
            appendInf = True
            Tmax = None
            ntemps = ntemps - 1

        if ntemps is not None:
            if Tmax is None:
                # Determine Tmax from ntemps.
                Tmax = tstep ** (ntemps - 1)
        else:
            if Tmax is None:
                raise ValueError('Must specify at least one of ``ntemps'' and '
                                'finite ``Tmax``.')

            # Determine ntemps from Tmax.
            ntemps = int(np.log(Tmax) / np.log(tstep) + 2)

        betas = np.logspace(0, -np.log10(Tmax), ntemps)
        if appendInf:
            # Use a geometric spacing, but replace the top-most temperature with
            # infinity.
            betas = np.concatenate((betas, [0]))

        return betas
        
        
    def assign_temperatures(self):
        # #Linear Spacing
        # temp = 2
        # for i in range(0,self.num_chains):
        #   self.temperatures.append(temp)
        #   temp += 2.5 #(self.maxtemp/self.num_chains)
        #   print (self.temperatures[i])
        #Geometric Spacing

        if self.geometric == True:
            betas = self.default_beta_ladder(2, ntemps=self.num_chains, Tmax=self.maxtemp)      
            for i in range(0, self.num_chains):         
                self.temperatures.append(np.inf if betas[i] is 0 else 1.0/betas[i])
                print (self.temperatures[i])
        else:

            tmpr_rate = (self.maxtemp /self.num_chains)
            temp = 1
            print("Temperatures...")
            for i in range(0, self.num_chains):            
                self.temperatures.append(temp)
                temp += tmpr_rate
                print(self.temperatures[i])


    
    def initialize_chains (self,      check_likelihood_sed,   burn_in):
        self.burn_in = burn_in
        self.vec_parameters =   np.random.uniform(self.minlimits_vec, self.maxlimits_vec) # will begin from diff position in each replica (comment if not needed)
        self.assign_temperatures()
        
        for i in range(0, self.num_chains):
            self.chains.append(ptReplica(  self.num_param, self.vec_parameters,  self.minlimits_vec, self.maxlimits_vec, self.stepratio_vec,  check_likelihood_sed ,self.swap_interval, self.sim_interval,   self.simtime, self.NumSamples, self.real_elev, 
               self.real_erodep_pts, self.erodep_coords, self.folder, self.xmlinput,  self.run_nb,self.temperatures[i], self.parameter_queue[i],self.event[i], self.wait_chain[i],burn_in, 
               self.surrogate_parameter_queues[i],self.surrogate_interval,
            self.surrogate_prob,self.surrogate_start_events[i],self.surrogate_resume_events[i], 
            self.save_surrogate_data,self.use_surrogate, self.compare_surrogate, self.pause_chain_events[i], self.resume_chain_events[i], self.surrogate_topology))
                    
             

    def swap_procedure(self, parameter_queue_1, parameter_queue_2):
        # if parameter_queue_2.empty() is False and parameter_queue_1.empty() is False:
        swapped = False
        param1 = parameter_queue_1.get()
        param2 = parameter_queue_2.get()
        w1 = param1[0:self.num_param]
        eta1 = param1[self.num_param]
        lhood1 = param1[self.num_param+1]
        T1 = param1[self.num_param+2]
        w2 = param2[0:self.num_param]
        eta2 = param2[self.num_param]
        lhood2 = param2[self.num_param+1]
        T2 = param2[self.num_param+2]
        #SWAPPING PROBABILITIES
        try:
            swap_proposal =  min(1,0.5*np.exp(min(709, lhood2 - lhood1)))
        except OverflowError:
            swap_proposal = 1
        u = np.random.uniform(0,1)
        if u < swap_proposal:
            self.num_swap += 1
            param_temp =  param1
            param1 = param2
            param2 = param_temp
            swapped = True
        else:
            swapped = False
        self.total_swap_proposals += 1
        print("swapped: {} {}".format(param1[:2], param2[:2]))
        return param1, param2, swapped

    def surrogate_trainer(self,params): 

        X = params[:,:self.num_param]
        Y = params[:,self.num_param].reshape(X.shape[0],1)
 

        for i in range(Y.shape[1]):
            min_Y = min(Y[:,i])
            max_Y = max(Y[:,i])
            self.minY[0,i] =   min_Y * 2
            self.maxY[0,i] = -1#max_Y

        self.model_signature += 1.0
        if self.model_signature == 1.0:
            np.savetxt(self.folder+'/surrogate/minmax.txt',[self.minY[0, 0], self.maxY[0, 0]])

        np.savetxt(self.folder+'/surrogate/model_signature.txt', [self.model_signature])

        Y= self.normalize_likelihood(Y)
        indices = np.where(Y==np.inf)[0]
        X = np.delete(X, indices, axis=0)
        Y = np.delete(Y,indices, axis=0)
        surrogate_model = surrogate("krnn", X , Y , self.minlimits_vec, self.maxlimits_vec, self.minY, self.maxY, self.folder, self.save_surrogate_data, self.surrogate_topology )
        surrogate_model.train(self.model_signature)



 

    def normalize_likelihood(self, Y):
        for i in range(Y.shape[1]):
            if self.model_signature == 1.0:
                min_Y = min(Y[:,i])
                max_Y = max(Y[:,i])
                # self.minY[0,i] = 1 #For Tau Squared
                # self.maxY[0,i] = max_Y


                # min -115 and max -96
                self.maxY[0,i] = -1 #max_Y
                self.minY[0,i] =  min_Y * 2

            # Y[:,i] = ([:,i] - min_Y)/(max_Y - min_Y)

            Y[:,i] = (Y[:,i] - self.minY[0,0])/(self.maxY[0,0]-self.minY[0,0])

        return Y

    def surr_procedure(self,queue): 
        if queue.empty() is False:
            return queue.get()
        else:
            return


    def run_chains(self):
        # only adjacent chains can be swapped therefore, the number of proposals is ONE less num_chains
        swap_proposal = np.ones(self.num_chains-1)
        # create parameter holders for paramaters that will be swapped
        replica_param = np.zeros((self.num_chains, self.num_param))
        lhood = np.zeros(self.num_chains)
        # Define the starting and ending of MCMC Chains
        start = 0
        end = self.NumSamples-1
        number_exchange = np.zeros(self.num_chains)
        filen = open(self.folder + '/num_exchange.txt', 'a')
        #RUN MCMC CHAINS
        for l in range(0,self.num_chains):
            self.chains[l].start_chain = start
            self.chains[l].end = end
        for j in range(0,self.num_chains):
            self.pause_chain_events[j].clear()
            self.resume_chain_events[j].clear()
            self.chains[j].start()
        swaps_appected_main = 0
        total_swaps_main = 0

        
 

        #SWAP PROCEDURE
        # while True:
        for i in range(int(self.NumSamples/self.surrogate_interval)):
            # Check if individual processes are still alive
            count = 0
            for index in range(self.num_chains):
                if not self.chains[index].is_alive():
                    count+=1
                    self.pause_chain_events[index].set()
                    # print(str(self.chains[index].temperature) +" Dead")
                # else:
                #     print(str(self.chains[index].temperature) +" Alive")
            if count == self.num_chains:
                break
            print("Waiting for swap signal.")

            # Check for signal from individual chains for swap
            signal_count = 0
            for index in range(0,self.num_chains):
                print("Waiting for chain: {}. Chain alive: {}".format(index+1, self.chains[index].is_alive()))
                flag = self.pause_chain_events[index].wait()
                if flag:
                    print("Signal from chain: {}".format(index+1))
                    # self.pause_chain_events[index].clear()
                    signal_count += 1

            # If signal not recieved from all chains skip the swap
            if signal_count == self.num_chains:

                all_param =   np.empty((1,self.num_param+1))
 



                for index in range(0,self.num_chains):
                    print('starting surr')
                    queue_surr=  self.surrogate_parameter_queues[index] 

                    surr_data = queue_surr.get() 

                    all_param =   np.concatenate([all_param,surr_data],axis=0) 
 

                data_train = all_param[1:,:]  
 

                self.surrogate_trainer(data_train) 

 
                for index in range(self.num_chains):
                    self.resume_chain_events[index].set()
                    self.pause_chain_events[index].clear() 


            elif signal_count == 0:
                break
            else:
                print("Skipping the action!")


        #JOIN THEM TO MAIN PROCESS
        for j in range(0,self.num_chains):
            self.chains[j].join()
        self.chain_queue.join()
        for i in range(0,self.num_chains):
            #self.parameter_queue[i].close()
            #self.parameter_queue[i].join_thread()
            self.surrogate_parameter_queues[i].close()
            self.surrogate_parameter_queues[i].join_thread()
     




        print(number_exchange, 'num_exchange, process ended')

        combined_topo,    accept, pred_topo, combined_topo  = self.show_results('chain_')

        
        for i in range(self.sim_interval.size):

            self.viewGrid(width=1000, height=1000, zmin=None, zmax=None, zData=combined_topo[i,:,:], title='Predicted Topography ', time_frame=self.sim_interval[i],  filename= 'mean')

        
        swap_perc = self.num_swap#*100/self.total_swap_proposals  

        


        return (pred_topo, swap_perc, accept)

 

    # Merge different MCMC chains y stacking them on top of each other
    def show_results(self, filename):

        burnin = int(self.NumSamples * self.burn_in)
        accept_percent = np.zeros((self.num_chains, 1)) 
        topo  = self.real_elev
        replica_topo = np.zeros((self.sim_interval.size, self.num_chains, topo.shape[0], topo.shape[1])) #3D
        combined_topo = np.zeros(( self.sim_interval.size, topo.shape[0], topo.shape[1]))
          
         

        for i in range(self.num_chains):
            for j in range(self.sim_interval.size):

                file_name = self.folder+'/posterior/predicted_topo/topo/chain_'+str(self.sim_interval[j])+'_'+ str(self.temperatures[i])+ '.txt'
                dat_topo = np.loadtxt(file_name)
                replica_topo[j,i,:,:] = dat_topo

                

        for j in range(self.sim_interval.size):
            for i in range(self.num_chains):
                combined_topo[j,:,:] += replica_topo[j,i,:,:]  
            combined_topo[j,:,:] = combined_topo[j,:,:]/self.num_chains

      


        accept = 0

        pred_topofinal = combined_topo[-1,:,:] # get the last mean pedicted topo to calculate mean squared error loss 
 

        return  combined_topo,    accept, pred_topofinal, combined_topo


 

        #---------------------------------------
        



    

    def viewGrid(self, width=1000, height=1000, zmin=None, zmax=None, zData=None, title='Predicted Topography', time_frame=None, filename=None):

          
        
        filename= self.folder +  '/pred_plots'+ '/pred_'+filename+'_'+str(time_frame)+ '_.png'

        fig = plt.figure()
        im = plt.imshow(zData, cmap='hot', interpolation='nearest')
        plt.colorbar(im)
        plt.savefig(filename)
        plt.close()

 
 


def make_directory (directory): 
    if not os.path.exists(directory):
        os.makedirs(directory)

 


def main():

    random.seed(time.time()) 
 

     

    if problem == 1: #this is CM-extended
        problemfolder = 'Examples/etopo/'
        xmlinput = problemfolder + 'etopo.xml'



        datapath = problemfolder + 'data/final_elev.txt'
        groundtruth_elev = np.loadtxt(datapath)
        groundtruth_erodep = np.loadtxt(problemfolder + 'data/final_erdp.txt')
        groundtruth_erodep_pts = np.loadtxt(problemfolder + 'data/final_erdp_pts.txt')


        res_summaryfile = '/results_canonicalPTbayeslands.txt'


        inittopo_expertknow = [] # no expert knowledge as simulated init topo


        simtime = 1000000
        resolu_factor = 1

        true_parameter_vec = np.loadtxt(problemfolder + 'data/true_values.txt')
        likelihood_sediment = True


        len_grid = 1  # ignore - this is in case if init topo is inferenced
        wid_grid = 1   # ignore


        real_rain = 1.5 #m/a
        real_erod = 5.e-6 
        m = 0.5  #Stream flow parameters
        n = 1 #
        real_cmarine = 5.e-1 # Marine diffusion coefficient [m2/a] -->
        real_caerial = 8.e-1 #aerial diffusion

        rain_min = 0.0
        rain_max = 3.0 

        # assume 4 regions and 4 time scales


        rain_regiongrid = 1  # how many regions in grid format 
        rain_timescale = 1  # to show climate change 
 


        minlimits_vec = [0, 4.e-6, 0, 0, 0,0]
        maxlimits_vec = [3, 6.e-6, 1, 2, 1,1]


 
        vec_parameters = np.random.uniform(minlimits_vec, maxlimits_vec) #  draw intial values for each of the free parameters
    
        stepsize_ratio  = 0.05 #   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now

        stepratio_vec =  np.repeat(stepsize_ratio, vec_parameters.size) 
        num_param = vec_parameters.size
 

        erodep_coords = np.array([[42,10],[39,8],[75,51],[59,13],[40,5],[6,20],[14,66],[4,40],[72,73],[46,64]])  # need to hand pick given your problem

        if (true_parameter_vec.shape[0] != vec_parameters.size ) :
            print( 'vec_params != true_values.txt ',true_parameter_vec.shape,vec_parameters.size)
            print( 'make sure that this is updated in case when you intro more parameters. should have as many rows as parameters ') 
            
            return

  
    elif problem == 2:
        problemfolder = 'Examples/mountain/'
        xmlinput = problemfolder + 'mountain.xml'
        simtime = 1000000
        resolu_factor = 1
        true_parameter_vec = np.loadtxt(problemfolder + 'data/true_values.txt')
        likelihood_sediment = True



        datapath = problemfolder + 'data/final_elev.txt'
        groundtruth_elev = np.loadtxt(datapath)
        groundtruth_erodep = np.loadtxt(problemfolder + 'data/final_erdp.txt')
        groundtruth_erodep_pts = np.loadtxt(problemfolder + 'data/final_erdp_pts.txt')



        #Set variables
        m = 0.5
        m_min = 0.
        m_max = 2
        
        n = 1.
        n_min = 0.
        n_max = 2.

        rain_real = 1.5
        rain_min = 0.
        rain_max = 3.

        erod_real = 5 
        erod_min = 3.e-6
        erod_max = 7.e-6
        #uplift_real = 50000
        uplift_min = 0.1 # X uplift_real
        uplift_max = 5.0 # X uplift_real
                
        #Rainfall, erodibility, m, n, uplift
        minlimits_vec=[rain_min,erod_min,m_min,n_min,uplift_min]
        maxlimits_vec=[rain_max,erod_max,m_max,n_max,uplift_max]
                
        vec_parameters = np.random.uniform(minlimits_vec, maxlimits_vec) #  draw intial values for each of the free parameters

        stepsize_ratio  = 0.05#   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now

        stepratio_vec =  np.repeat(stepsize_ratio, vec_parameters.size) 
        #stepratio_vec = [stepsize_ratio, stepsize_ratio, stepsize_ratio, stepsize_ratio, 0.02] 
        #stepratio_vec = [0.1, 0.1, 0.1, 0.1, 0.1]
        print("steps: ", stepratio_vec)
        num_param = vec_parameters.size
        erodep_coords=np.array([[5,5],[10,10],[20,20],[30,30],[40,40],[50,50],[25,25],[37,30],[44,27],[46,10]])

        if (true_parameter_vec.shape[0] != vec_parameters.size ) :
            print( 'vec_params != true_values.txt ',true_parameter_vec.shape,vec_parameters.size)
            print( 'make sure that this is updated in case when you intro more parameters. should have as many rows as parameters ') 
            
            return



    elif problem == 3:
        problemfolder = 'Examples/tasmania/'
        xmlinput = problemfolder + 'tasmania.xml'
        simtime = 1000000
        resolu_factor = 1

        true_parameter_vec = np.loadtxt(problemfolder + 'data/true_values.txt')



        datapath = problemfolder + 'data/final_elev.txt'
        groundtruth_elev = np.loadtxt(datapath)
        groundtruth_erodep = np.loadtxt(problemfolder + 'data/final_erdp.txt')
        groundtruth_erodep_pts = np.loadtxt(problemfolder + 'data/final_erdp_pts.txt')



        inittopo_expertknow = [] # no expert knowledge as simulated init topo

        m = 0.5 # used to be constants  
        n = 1

        real_rain = 1.5
        real_erod = 5.e-5

        likelihood_sediment = True

        real_caerial = 8.e-1 
        real_cmarine = 5.e-1 # Marine diffusion coefficient [m2/a] -->

        maxlimits_vec = [3.0,7.e-6, 2, 2,  1.0, 0.7]  # [rain, erod] this can be made into larger vector, with region based rainfall, or addition of other parameters
        minlimits_vec = [0.0 ,3.e-6, 0, 0, 0.6, 0.3 ]   # hence, for 4 regions of rain and erod[rain_reg1, rain_reg2, rain_reg3, rain_reg4, erod_reg1, erod_reg2, erod_reg3, erod_reg4 ]
                                    ## hence, for 4 regions of rain and 1 erod, plus other free parameters (p1, p2) [rain_reg1, rain_reg2, rain_reg3, rain_reg4, erod, p1, p2 ]
                                    #if you want to freeze a parameter, keep max and min limits the same


 
        vec_parameters = np.random.uniform(minlimits_vec, maxlimits_vec) #  draw intial values for each of the free parameters
    
        stepsize_ratio  = 0.05#   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now

        stepratio_vec =  np.repeat(stepsize_ratio, vec_parameters.size) 
        num_param = vec_parameters.size
         

        erodep_coords = np.array([[42,10],[39,8],[75,51],[59,13],[40,5],[6,20],[14,66],[4,40],[72,73],[46,64]])  # need to hand pick given your problem
 


 

    else:
        print('choose some problem  ')

 


    fname = ""
    run_nb = 0
    while os.path.exists( problemfolder+ 'results_%s' % (run_nb)):
        run_nb += 1
    if not os.path.exists( problemfolder+ 'results_%s' % (run_nb)):
        os.makedirs( problemfolder+ 'results_%s' % (run_nb))
        fname = ( problemfolder+ 'results_%s' % (run_nb))

    #fname = ('sampleresults')

    make_directory((fname + '/posterior/pos_parameters')) 

    make_directory((fname + '/recons_initialtopo')) 

    make_directory((fname + '/pos_plots')) 
    make_directory((fname + '/posterior/predicted_topo/topo'))  

    make_directory((fname + '/posterior/predicted_topo/sed'))  

    make_directory((fname + '/posterior/predicted_topo/x_slice'))

    make_directory((fname + '/posterior/predicted_topo/y_slice'))


    make_directory((fname + '/posterior/surg_likelihood'))


 

    make_directory((fname + '/posterior/posterior/predicted_erodep')) 
    make_directory((fname + '/pred_plots'))


    make_directory((fname + '/performance/lhood'))
    make_directory((fname + '/performance/accept'))
    make_directory((fname + '/performance/rmse_edep'))
    make_directory((fname + '/performance/rmse_elev'))


    make_directory((fname + '/surrogate')) 
    make_directory((fname + '/surrogate/prediction_benchmark_data'))

    make_directory((fname + '/surrogate/learnsurrogate_data'))
 


    np.savetxt('foldername.txt', np.array([fname]), fmt="%s")



 
 

    run_nb_str =  'results_' + str(run_nb)
 

    timer_start = time.time()

    sim_interval = np.arange(0,  simtime+1, simtime/num_successive_topo) # for generating successive topography
    print("Simulation time interval", sim_interval)

    surrogate_interval = surrogate_int
    #surrogate_prob = surrogate_prob
    save_surrogatedata = True
    use_surrogate = True
    compare_surrogate = True

    #-------------------------------------------------------------------------------------
    #Create A a Patratellel Tempring object instance 
    #-------------------------------------------------------------------------------------
    pt = ParallelTempering(  vec_parameters, minlimits_vec, maxlimits_vec, stepratio_vec,   num_chains,
     maxtemp, samples,swap_interval,fname, true_parameter_vec, num_param  ,  groundtruth_elev,  groundtruth_erodep_pts , erodep_coords, 
     simtime, sim_interval, resolu_factor, run_nb_str, xmlinput, surrogate_interval, surrogate_prob, save_surrogatedata, use_surrogate, compare_surrogate)
    
    #-------------------------------------------------------------------------------------
    # intialize the MCMC chains
    #-------------------------------------------------------------------------------------
    pt.initialize_chains(     likelihood_sediment,   burn_in)

    #-------------------------------------------------------------------------------------
    #run the chains in a sequence in ascending order
    #-------------------------------------------------------------------------------------
    pred_topofinal, swap_perc, accept  = pt.run_chains()



    print('sucessfully sampled') 
    timer_end = time.time()  

    elapsed = timer_end - timer_start



    res_file = open('results_surrogaterevamp_'+str(problem)+'.txt',"ab") 
    np.savetxt(res_file,   np.array([surrogate_int, surrogate_prob, elapsed]), fmt='%1.2f'  ) 


    np.savetxt(fname+'/res_time.txt',  np.array([surrogate_int, surrogate_prob, elapsed]), fmt='%1.4f'  )   


    res_file.close()

    '''dir_name = fname + '/posterior'
    fname_remove = fname +'/pos_param.txt'
    print(dir_name)
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)

    if os.path.exists(fname_remove):  # comment if you wish to keep pos file
        os.remove(fname_remove)'''



    #stop()
if __name__ == "__main__": main()
