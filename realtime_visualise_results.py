

#Main Contributers:   Rohitash Chandra and Ratneel Deo  Email: c.rohitash@gmail.com, deo.ratneel@gmail.com

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


from badlands.model import Model as badlandsModel
import badlands 



from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import itertools 
import pandas
import argparse

import pandas as pd
import seaborn as sns


  
from scipy.ndimage import filters 

import scipy.ndimage as ndimage

from scipy.ndimage import gaussian_filter

#Initialise and parse inputs
parser=argparse.ArgumentParser(description='PTBayeslands modelling')

parser.add_argument('-p','--problem', help='Problem Number 1-crater-fast,2-crater,3-etopo-fast,4-etopo,5-null,6-mountain', required=True,   dest="problem",type=int)
parser.add_argument('-s','--samples', help='Number of samples', default=10000, dest="samples",type=int)
parser.add_argument('-r','--replicas', help='Number of chains/replicas, best to have one per availble core/cpu', default=10,dest="num_chains",type=int)
parser.add_argument('-t','--temperature', help='Demoninator to determine Max Temperature of chains (MT=no.chains*t) ', default=10,dest="mt_val",type=int)
parser.add_argument('-swap','--swap', help='Swap Ratio', dest="swap_ratio",default=0.02,type=float)
parser.add_argument('-b','--burn', help='How many samples to discard before determing posteriors', dest="burn_in",default=0.25,type=float)
parser.add_argument('-pt','--ptsamples', help='Ratio of PT vs straight MCMC samples to run', dest="pt_samples",default=0.5,type=float)  
parser.add_argument('-rain_intervals','--rain_intervals', help='rain_intervals', dest="rain_intervals",default=4,type=int)


parser.add_argument('-epsilon','--epsilon', help='epsilon for inital topo', dest="epsilon",default=0.5,type=float)



args = parser.parse_args()
    
#parameters for Parallel Tempering
problem = args.problem
samples = args.samples #10000  # total number of samples by all the chains (replicas) in parallel tempering
num_chains = args.num_chains
swap_ratio = args.swap_ratio
burn_in=args.burn_in
#maxtemp = int(num_chains * 5)/args.mt_val
maxtemp =   args.mt_val 
swap_interval = int(swap_ratio * (samples/num_chains)) #how ofen you swap neighbours
num_successive_topo = 4
pt_samples = args.pt_samples
epsilon = args.epsilon
rain_intervals = args.rain_intervals

method = 1 # type of formaltion for inittopo construction (Method 1 showed better results than Method 2)

class results_visualisation:

    def __init__(self, vec_parameters,   num_chains, maxtemp, samples,swap_interval,fname, num_param  ,  groundtruth_elev,  groundtruth_erodep_pts , erodep_coords, simtime, sim_interval, resolu_factor,  xmlinput,  run_nb_str ):

   
        self.swap_interval = swap_interval
        self.folder = fname
        self.maxtemp = maxtemp
        self.num_swap = 0
        self.num_chains = num_chains
        self.chains = []
        self.temperatures = []
        self.NumSamples = samples
        self.sub_sample_size = max(1, int( 0.05* self.NumSamples))
        self.show_fulluncertainity = False # needed in cases when you reall want to see full prediction of 5th and 95th percentile of topo. takes more space 
        self.real_erodep_pts  = groundtruth_erodep_pts
        self.real_elev = groundtruth_elev
        self.resolu_factor =  resolu_factor
        self.num_param = num_param
        self.erodep_coords = erodep_coords
        self.simtime = simtime
        self.sim_interval = sim_interval
        #self.run_nb =run_nb 
        self.xmlinput = xmlinput
        self.run_nb_str =  run_nb_str
        self.vec_parameters = vec_parameters
        #self.realvalues  =  realvalues_vec 

        self.burn_in = burn_in


        self.minY = np.zeros((1,1))
        self.maxY = np.ones((1,1)) 


        
        # create queues for transfer of parameters between process chain
        #self.chain_parameters = [multiprocessing.Queue() for i in range(0, self.num_chains) ]
        self.parameter_queue = [multiprocessing.Queue() for i in range(num_chains)]
        self.chain_queue = multiprocessing.JoinableQueue()  
        self.wait_chain = [multiprocessing.Event() for i in range (self.num_chains)]

        # two ways events are used to synchronize chains
        self.event = [multiprocessing.Event() for i in range (self.num_chains)]
        #self.wait_chain = [multiprocessing.Event() for i in range (self.num_chains)]

        self.geometric =  True
        self.total_swap_proposals = 0
 

        self.use_surrogate = True 




    def  results_current (self ):
         

        #pos_param, likelihood_rep, accept_list, pred_topo,  combined_erodep, accept, pred_topofinal, list_xslice, list_yslice, rmse_elev, rmse_erodep = self.show_results('chain_')

        posterior, likelihood_vec, accept_list,   xslice, yslice, rmse_elev, rmse_erodep, erodep_pts, rmse_surrogate = self.show_results('chain_')


        self.view_crosssection_uncertainity(xslice, yslice)

        optimal_para, para_5thperc, para_95thperc = self.get_uncertainity(likelihood_vec, posterior)
        np.savetxt(self.folder+'/optimal_percentile_para.txt', np.array([optimal_para, para_5thperc, para_95thperc]) )


        for s in range(self.num_param):  
            self.plot_figure(posterior[s,:], 'pos_distri_'+str(s) ) 

       
  
 
        mean_pos = posterior.mean(axis=1) 

        percentile_95th = np.percentile(posterior, 95, axis=1) 

        percentile_5th = np.percentile(posterior, 5, axis=1) 
 
 


        #return (pos_param,likelihood_rep, accept_list,   combined_erodep,  pred_topofinal, swap_perc, accept,  rmse_elev, rmse_erodep, rmse_slice_init, rmse_full_init)
        return  posterior, likelihood_vec, accept_list,   xslice, yslice, rmse_elev, rmse_erodep, erodep_pts,  rmse_surrogate


     


    def view_crosssection_uncertainity(self,  list_xslice, list_yslice):
        print ('list_xslice', list_xslice.shape)
        print ('list_yslice', list_yslice.shape)

        ymid = int(self.real_elev.shape[1]/2 ) #   cut the slice in the middle 
        xmid = int(self.real_elev.shape[0]/2)

        print( 'ymid',ymid)
        print( 'xmid', xmid)
        print(self.real_elev)
        print(self.real_elev.shape, ' shape')

        x_ymid_real = self.real_elev[xmid, :] 
        y_xmid_real = self.real_elev[:, ymid ] 
        x_ymid_mean = list_xslice.mean(axis=1)

        print( x_ymid_real.shape , ' x_ymid_real shape')
        print( x_ymid_mean.shape , ' x_ymid_mean shape')
        
        x_ymid_5th = np.percentile(list_xslice, 5, axis=1)
        x_ymid_95th= np.percentile(list_xslice, 95, axis=1)

        y_xmid_mean = list_yslice.mean(axis=1)
        y_xmid_5th = np.percentile(list_yslice, 5, axis=1)
        y_xmid_95th= np.percentile(list_yslice, 95, axis=1)


        x = np.linspace(0, x_ymid_mean.size * self.resolu_factor, num=x_ymid_mean.size) 
        x_ = np.linspace(0, y_xmid_mean.size * self.resolu_factor, num=y_xmid_mean.size)

        #ax.set_xlim(-width,len(ind)+width)

        self.cross_section(x, x_ymid_mean, x_ymid_real, x_ymid_5th, x_ymid_95th, 'x_ymid_cross')
        self.cross_section(x_, y_xmid_mean, y_xmid_real, y_xmid_5th, y_xmid_95th, 'y_xmid_cross')


     
    def cross_section(self, x, pred, real, lower, higher, fname):

        size = 15



        ticksize = 14

        fig = plt.figure()
        ax = fig.add_subplot(111)
        #index = np.arange(groundtruth_erodep_pts.size) 
        #ground_erodepstd = np.zeros(groundtruth_erodep_pts.size) 
        opacity = 0.8
        width = 0.35       # the width of the bars


        rmse_init = np.sqrt(np.sum(np.square(pred  -  real))  / real.size) 

        ax.plot(x,  real, label='Ground-truth') 
        ax.plot(x, pred, label='Badlands pred.') 


 
        #plotlegend = ax.legend( (rects1[0], rects2[0]), ('Predicted  ', ' Ground-truth ') )
      

        ax.fill_between(x, lower , higher, facecolor='g', alpha=0.2, label = 'Uncertainty')
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.legend(loc='best', fontsize=12)  

        ax.set_ylabel('Height in meters', fontsize=ticksize-1)
        ax.set_xlabel(' Distance (km) ', fontsize=ticksize-1)
        #ax.set_title(' Topography  cross section', fontsize=ticksize)
    
        ax.grid(alpha=0.75)
 

        ax.tick_params(labelsize=ticksize)

        plt.tight_layout()
    
    
        plt.savefig(self.folder+'/'+fname+'.pdf')
        plt.clf()   

        return rmse_init


 


    def get_synthetic_initopo(self):

        model = badlandsModel() 
        # Load the XmL input file
        model.load_xml(str(self.run_nb_str), self.xmlinput, muted=True) 
        #Update the initial topography
        #Use the coordinates from the original dem file
        xi=int(np.shape(model.recGrid.rectX)[0]/model.recGrid.nx)
        yi=int(np.shape(model.recGrid.rectY)[0]/model.recGrid.ny)
        #And put the demfile on a grid we can manipulate easily
        elev=np.reshape(model.recGrid.rectZ,(xi,yi))

        return elev

    def normalize_likelihood(self, Y):
        for i in range(Y.shape[1]):
            if self.model_signature == 1.0:
                min_Y = min(Y[:,i])
                max_Y = max(Y[:,i])
                self.maxY[0,i] = max_Y
                self.minY[0,i] = min_Y
            
            Y[:,i] = (Y[:,i] - self.minY[0,0])/(self.maxY[0,0]-self.minY[0,0])

        return Y


    # Merge different MCMC chains y stacking them on top of each other
    def show_results(self, filename):

        

        path = self.folder +'/posterior/pos_parameters/' 
        x = [] # first get the size of the files

        files = os.listdir(path)
        for name in files: 
            dat = np.loadtxt(path+name)
            x.append(dat.shape[0])
            print(dat.shape) 

        print(x)
        size_pos = min(x) 

        self.num_chains = len(x)


        print(len(x), size_pos, self.num_chains,    ' ***')
        self.NumSamples = int((self.num_chains * size_pos)/ self.num_chains) -2


        print(self.NumSamples,    ' ***')



        burnin =  int((self.NumSamples * self.burn_in)/self.num_chains)

        #if burnin == size_pos:



        coverage = self.NumSamples - burnin

        pos_param = np.zeros((self.num_chains, self.NumSamples  , self.num_param))
        list_xslice = np.zeros((self.num_chains, self.NumSamples , self.real_elev.shape[1]))
        list_yslice = np.zeros((self.num_chains, self.NumSamples  , self.real_elev.shape[0] ))
        likehood_rep = np.zeros((self.num_chains, self.NumSamples))

        surrogate_lhood = np.zeros(( self.num_chains, self.NumSamples, 3))



         # index 1 for likelihood posterior and index 0 for Likelihood proposals. Note all likilihood proposals plotted only
        #accept_percent = np.zeros((self.num_chains, 1))
        accept_list = np.zeros((self.num_chains, self.NumSamples )) 
        topo  = self.real_elev
        #replica_topo = np.zeros((self.sim_interval.size, self.num_chains, topo.shape[0], topo.shape[1])) #3D
        #combined_topo = np.zeros(( self.sim_interval.size, topo.shape[0], topo.shape[1]))

        edp_pts_time = self.real_erodep_pts.shape[1] *self.sim_interval.size

        erodep_pts = np.zeros(( self.num_chains, self.NumSamples  , edp_pts_time )) 
        combined_erodep = np.zeros((self.num_chains, self.NumSamples, self.real_erodep_pts.shape[1] ))
        timespan_erodep = np.zeros(( (self.NumSamples - burnin) * self.num_chains, self.real_erodep_pts.shape[1] ))
        rmse_elev = np.zeros((self.num_chains, self.NumSamples))
        rmse_erodep = np.zeros((self.num_chains, self.NumSamples))




        print(self.NumSamples, size_pos, burnin, ' self.NumSamples, size_pos, burn')



        path = self.folder +'/posterior/pos_parameters/' 
        files = os.listdir(path)
        v = 0 
        for name in files: 
            dat = np.loadtxt(path+name) 
            print(dat.shape, pos_param.shape,  v, burnin, size_pos, coverage) 
            pos_param[v, :, :] = dat[ :pos_param.shape[1],:] 
            #print (dat)
            print(v, name, ' is v')
            v = v +1 

        posterior = pos_param.transpose(2,0,1).reshape(self.num_param,-1)  
 

        path = self.folder +'/posterior/predicted_topo/x_slice/' 
        files = os.listdir(path)
        v = 0 
        for name in files: 
            dat = np.loadtxt(path+name) 
            list_xslice[v, :, :] = dat[ : list_xslice.shape[1],: ] 
            v = v +1


        list_xslice = list_xslice[:, burnin:, :]

        xslice = list_xslice.transpose(2,0,1).reshape(self.real_elev.shape[1],-1) 

        print(list_xslice.shape, xslice.shape, self.real_elev.shape,  'list_xslice size') 



 

        path = self.folder +'/posterior/predicted_topo/y_slice/' 
        files = os.listdir(path)
        v = 0 
        for name in files: 
            dat = np.loadtxt(path+name) 
            list_yslice[v, :, :] = dat[ : list_yslice.shape[1],: ] 
            v = v +1 

        list_yslice = list_yslice[:, burnin:, :] 
        yslice = list_yslice.transpose(2,0,1).reshape(self.real_elev.shape[0],-1) 




        path = self.folder +'/posterior/predicted_topo/sed/' 
        files = os.listdir(path)
        v = 0 
        for name in files: 
            dat = np.loadtxt(path+name) 
            erodep_pts[v, :, :] = dat[ : erodep_pts.shape[1],: ] 
            v = v +1 

        erodep_pts = erodep_pts[:, burnin:, :] 
        
        erodep_pts = erodep_pts.transpose(2,0,1).reshape(edp_pts_time,-1) 
        print(erodep_pts.shape, ' ed   ***')
 
 




 

        path = self.folder +'/performance/lhood/' 
        files = os.listdir(path)

        v = 0 
        for name in files: 
            dat = np.loadtxt(path+name) 
            likehood_rep[v, : ] = dat[ : likehood_rep.shape[1]] 
            v = v +1  


        path = self.folder +'/posterior/surg_likelihood/' 
        files = os.listdir(path)

        v = 0 
        for name in files: 
            print(name, ' lhood srg')
            dat = np.loadtxt(path+name) 
            surrogate_lhood[v, : ] = dat[ : surrogate_lhood.shape[1]] 

            #surrogate_lhood[v, :, :] = dat[ :surrogate_lhood.shape[1],:] 
            v = v +1  
 


        surrogate_combined = surrogate_lhood.transpose(2,0,1).reshape(3,-1)  

        #print(surrogate_combined, ' surrogate_comb')


 
        path = self.folder +'/performance/accept/' 
        files = os.listdir(path)

        v = 0 
        for name in files: 
            dat = np.loadtxt(path+name) 
            accept_list[v, : ] = dat[ : accept_list.shape[1]] 
            v = v +1 
        #accept_list = accept_list[:, burnin: ] 

        path = self.folder +'/performance/rmse_edep/' 
        files = os.listdir(path) 
        v = 0 
        for name in files: 
            dat = np.loadtxt(path+name)  
            rmse_erodep[v, :  rmse_erodep.shape[1] ] = dat[ : rmse_erodep.shape[1]] 
            v = v +1 
        rmse_erodep = rmse_erodep[:, burnin: ] 


        path = self.folder +'/performance/rmse_elev/'
        files = os.listdir(path)
 

        v = 0 
        for name in files: 
            dat = np.loadtxt(path+name) 
            rmse_elev[v, : ] = dat[ : rmse_elev.shape[1]] 
            v = v +1 
        rmse_elev = rmse_elev[:, burnin: ]

 


        likelihood_vec = likehood_rep 
        accept_list = accept_list 




        rmse_elev = rmse_elev.reshape(self.num_chains*(self.NumSamples -burnin ),1)
 

        rmse_erodep = rmse_erodep.reshape(self.num_chains*(self.NumSamples -burnin  ),1) 
 

        np.savetxt(self.folder + '/pos_param.txt', posterior.T) 
        np.savetxt(self.folder + '/likelihood.txt', likelihood_vec.T, fmt='%1.5f')
        np.savetxt(self.folder + '/accept_list.txt', accept_list, fmt='%1.2f')
        #np.savetxt(self.folder + '/acceptpercent.txt', [accept], fmt='%1.2f')

        rmse_surrogate = 0 


        if self.use_surrogate is True:

            surrogate_likl = surrogate_combined.T
            surrogate_likl = surrogate_likl[~np.isnan(surrogate_likl).any(axis=1)]
            # surrogate_likl = surrogate_likl[~np.isinf(surrogate_likl).any(axis=1)]

            slen = np.arange(0,surrogate_likl.shape[0],1)
            fig = plt.figure(figsize = (12,12))
            ax = fig.add_subplot(111) 
            plt.tick_params(labelsize=20)

            params = {'legend.fontsize': 20, 'legend.handlelength': 2}
            plt.rcParams.update(params)
            surrogate_plot = ax.plot(slen,surrogate_likl[:,1],linestyle='-', linewidth= 1, color= 'b', label= 'Surrogate ')
            model_plot = ax.plot(slen,surrogate_likl[:,0],linestyle= '--', linewidth = 1, color = 'k', label = 'True')
            # residual_plot = ax.plot(slen,surrogate_likl[:,1]- surrogate_likl[:,0],linestyle= '-', linewidth = 1, color = 'r', label = 'True')
            
            ax.set_xlabel('Samples per Replica [R-1, R-2 ..., R-N] ',size= 20)
            ax.set_ylabel(' Log-Likelihood', size= 20)
            # ax.set_xlim([0,np.amax(slen)]) 
            ax.legend(loc='best')
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            plt.savefig('%s/surrogate_likl.png'% (self.folder), dpi=300, transparent=False)
            plt.clf()

            rmse_surrogate = np.sqrt(((surrogate_likl[:,1]-surrogate_likl[:,0])**2).mean()) 
 

      
        
        return posterior, likelihood_vec, accept_list,   xslice, yslice, rmse_elev, rmse_erodep, erodep_pts, rmse_surrogate


        
        #return posterior,    xslice, yslice 


    def find_nearest(self, array,value): # just to find nearest value of a percentile (5th or 9th from pos likelihood)
        idx = (np.abs(array-value)).argmin()
        return array[idx], idx

    def get_uncertainity(self, likehood_rep, pos_param ): 

        likelihood_pos = likehood_rep[:,1]

        a = np.percentile(likelihood_pos, 5)   
        lhood_5thpercentile, index_5th = self.find_nearest(likelihood_pos,a)  
        b = np.percentile(likelihood_pos, 95) 
        lhood_95thpercentile, index_95th = self.find_nearest(likelihood_pos,b)  
        max_index = np.argmax(likelihood_pos) # find max of pos liklihood to get the max or optimal pos value  

        optimal_para = pos_param[:, max_index] 
        para_5thperc = pos_param[:, index_5th]
        para_95thperc = pos_param[:, index_95th] 

        return optimal_para, para_5thperc, para_95thperc


     


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



    def plot_figure(self, list, title): 

        list_points =  list
        fname = self.folder
         


        size = 15

        plt.tick_params(labelsize=size)
        params = {'legend.fontsize': size, 'legend.handlelength': 2}
        plt.rcParams.update(params)
        plt.grid(alpha=0.75)

        plt.hist(list_points,  bins = 20, color='#0504aa',
                            alpha=0.7)   

        plt.title("Posterior distribution ", fontsize = size)
        plt.xlabel(' Parameter value  ', fontsize = size)
        plt.ylabel(' Frequency ', fontsize = size)
        plt.tight_layout()  
        plt.savefig(fname + '/pos_plots/' + title  + '_posterior.pdf')
        plt.clf()


        plt.tick_params(labelsize=size)
        params = {'legend.fontsize': size, 'legend.handlelength': 2}
        plt.rcParams.update(params)
        plt.grid(alpha=0.75)

        listx = np.asarray(np.split(list_points,  self.num_chains ))
        plt.plot(listx.T)   

        plt.title("Parameter trace plot", fontsize = size)
        plt.xlabel(' Number of Samples  ', fontsize = size)
        plt.ylabel(' Parameter value ', fontsize = size)
        plt.tight_layout()  
        plt.savefig(fname + '/pos_plots/' + title  + '_trace.pdf')
        plt.clf()


        #---------------------------------------
        



    

    def viewGrid(self, width=1000, height=1000, zmin=None, zmax=None, zData=None, title='Predicted Topography', time_frame=None, filename=None):

         
        
        filename= self.folder +  '/pred_plots'+ '/pred_'+filename+'_'+str(time_frame)+ '_.png'

        fig = plt.figure()
        im = plt.imshow(zData, cmap='hot', interpolation='nearest')
        plt.colorbar(im)
        plt.savefig(filename)
        plt.close()

 

def mean_sqerror(  pred_erodep,   real_erodep_pts):
        
        #elev = np.sqrt(np.sum(np.square(pred_elev -  real_elev))  / real_elev.size)  
        sed =  np.sqrt(  np.sum(np.square(pred_erodep -  real_erodep_pts)) / real_erodep_pts.size  ) 

        return   sed


def make_directory (directory): 
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_erodeposition(erodep_mean, erodep_std, groundtruth_erodep_pts, sim_interval, fname):


    ticksize = 14

    fig = plt.figure()
    ax = fig.add_subplot(111)
    index = np.arange(groundtruth_erodep_pts.size) 
    ground_erodepstd = np.zeros(groundtruth_erodep_pts.size) 
    opacity = 0.8
    width = 0.35       # the width of the bars

    rects1 = ax.bar(index, erodep_mean, width,
                color='blue',
                yerr=erodep_std,
                error_kw=dict(elinewidth=2,ecolor='red'))

    rects2 = ax.bar(index+width, groundtruth_erodep_pts, width, color='green', 
                yerr=ground_erodepstd,
                error_kw=dict(elinewidth=2,ecolor='red') )
 

    ax.set_ylabel('Height in meters', fontsize=ticksize)
    ax.set_xlabel('Location ID ', fontsize=ticksize)
    ax.set_title('Erosion/Deposition', fontsize=ticksize)
    
    ax.grid(alpha=0.75)

 
    ax.tick_params(labelsize=ticksize)
 
    plotlegend = ax.legend( (rects1[0], rects2[0]), ('Predicted  ', ' Ground-truth ') , fontsize=14 )
    plt.tight_layout()
    
    plt.savefig(fname +'/pos_erodep_'+str( sim_interval) +'_.pdf')
    plt.clf()    




def main():

    random.seed(time.time()) 



    if problem == 1: #this is CM-extended
        problemfolder = 'Examples/etopo/'
        xmlinput = problemfolder + 'etopo.xml'



        datapath = problemfolder + 'data/final_elev.txt'
        groundtruth_elev = np.loadtxt(datapath)
        groundtruth_erodep = np.loadtxt(problemfolder + 'data/final_erdp.txt')
        groundtruth_erodep_pts = np.loadtxt(problemfolder + 'data/final_erdp_pts.txt')



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

 


 
 


    #fname = np.genfromtxt('foldername.txt',dtype='str')


    with open ("foldername.txt", "r") as f:
        fname = f.read().splitlines() 

    fname = fname[0].rstrip()

    print(fname, ' fname -------------------')


    run_nb_str = fname

    timer_start = time.time()

    sim_interval = np.arange(0,  simtime+1, simtime/num_successive_topo) # for generating successive topography
    print("Simulation time interval", sim_interval)


    #-------------------------------------------------------------------------------------
     
    res = results_visualisation(  vec_parameters,  num_chains, maxtemp, samples,swap_interval,fname, num_param  ,  groundtruth_elev,  groundtruth_erodep_pts , erodep_coords, simtime, sim_interval, resolu_factor,  xmlinput,  run_nb_str)
    
    #------------------------------------------------------------------------------------- 
    pos_param, likehood_rep, accept_list,   xslice, yslice, rmse_elev, rmse_erodep, erodep_pts,  rmse_surrogate   = res.results_current()



    print('sucessfully sampled') 
    timer_end = time.time() 
    likelihood = likehood_rep # just plot proposed likelihood  
    #likelihood = np.asarray(np.split(likelihood,  num_chains ))

    plt.plot(likelihood.T)
    plt.savefig( fname+'/likelihood.pdf')
    plt.clf()

    size = 15 

    plt.tick_params(labelsize=size)
    params = {'legend.fontsize': size, 'legend.handlelength': 2}
    plt.rcParams.update(params)
    plt.plot(accept_list.T)
    plt.title("Replica Acceptance ", fontsize = size)
    plt.xlabel(' Number of Samples  ', fontsize = size)
    plt.ylabel(' Number Accepted ', fontsize = size)
    plt.tight_layout()
    plt.savefig( fname+'/accept_list.pdf' )
    plt.clf()

    print(erodep_pts.shape, ' erodep_pts.shape')

    #combined_erodep =   #np.reshape(erodep_pts, (3,-1)) 


 
    pred_erodep = np.zeros(( groundtruth_erodep_pts.shape[0], groundtruth_erodep_pts.shape[1] )) # just to get the right size


    for i in range(sim_interval.size): 

        begin = i * groundtruth_erodep_pts.shape[1] # number of points 
        end = begin + groundtruth_erodep_pts.shape[1] 

        pos_ed = erodep_pts[begin:end, :] 
        pos_ed = pos_ed.T 
        erodep_mean = pos_ed.mean(axis=0)  
        erodep_std = pos_ed.std(axis=0)  
        pred_erodep[i,:] = pos_ed.mean(axis=0)  

        print(erodep_mean, erodep_std, groundtruth_erodep_pts[i,:], sim_interval[i], fname) 
        plot_erodeposition(erodep_mean, erodep_std, groundtruth_erodep_pts[i,:], sim_interval[i], fname) 
        #np.savetxt(fname + '/posterior/predicted_erodep/com_erodep_'+str(sim_interval[i]) +'_.txt', pos_ed)

  

    pred_elev = np.array([])

    #rmse, rmse_sed= mean_sqerror(  pred_erodep, pred_elev,  groundtruth_elev,  groundtruth_erodep_pts)

    rmse_sed= mean_sqerror(  pred_erodep,  groundtruth_erodep_pts)

    rmse = 0

     

    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111)

    
    size = 15

    ax.tick_params(labelsize=size)

    plt.legend(loc='upper right') 

    ax.boxplot(pos_param.T) 
    ax.set_xlabel('Parameter ID', fontsize=size)
    ax.set_ylabel('Posterior', fontsize=size) 
    plt.title("Boxplot of Posterior", fontsize=size) 
    plt.savefig(fname+'/badlands_pos.pdf')
    
    #print (num_chains, problemfolder, run_nb_str, (timer_end-timer_start)/60, rmse_sed, rmse_elev)


    timer_end = time.time() 
    #likelihood = likehood_rep[:,0] # just plot proposed likelihood  
    #likelihood = np.asarray(np.split(likelihood,  num_chains ))

    rmse_el = np.mean(rmse_elev[:])
    rmse_el_std = np.std(rmse_elev[:])
    rmse_el_min = np.amin(rmse_elev[:])
    rmse_er = np.mean(rmse_erodep[:])
    rmse_er_std = np.std(rmse_erodep[:])
    rmse_er_min = np.amin(rmse_erodep[:])


    time_total = (timer_end-timer_start)/60

  
    res_file = open('results_surrogaterevamp_'+str(problem)+'.txt',"ab")  
 

    allres =  np.asarray([ problem, num_chains,  samples,   rmse_el, 
                        rmse_er, rmse_el_std, rmse_er_std, rmse_el_min, 
                        rmse_er_min, rmse, rmse_sed,    time_total, rmse_surrogate ]) 

    print(allres)


    np.savetxt(fname+'/res_file.txt',  np.array([allres]), fmt='%1.4f'  )  

 
    np.savetxt(res_file,  np.array([allres]), fmt='%1.4f'  )

    res_file.close()   
 
 
    dir_name = fname + '/posterior'
    #fname_remove = fname +'/pos_param.txt'
    #print(dir_name) 



    #stop()
if __name__ == "__main__": main()
