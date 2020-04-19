##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the BayesLands surface processes modelling companion.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
This script is intended to implement an MCMC (Markov Chain Monte Carlo) Metropolis Hastings methodology to pyBadlands. 
Badlands is used as a "black box" model for bayesian methods.
"""
import os
import numpy as np
import random
import time
import math
import copy
import shutil
import plotly
import collections
import plotly.plotly as py
import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from matplotlib.colors import LogNorm

import sklearn
from sklearn.model_selection import train_test_split

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import normalize
from copy import deepcopy
from io import StringIO
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import cKDTree
from scipy import stats 
from scipy.stats import multivariate_normal
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from plotly.graph_objs import *
from plotly.offline.offline import _plot_html
plotly.offline.init_notebook_mode()

class pt_surrogate_GP():
	"""
		
	"""
	def __init__(self, filename, run_nb):
		self.filename = filename
		self.run_nb = run_nb
	
	def viewGrid(self, x1Data, x2Data, yData, C, G, D, filename, title):
		# print x1Data.min(), x1Data.max()
		# print x2Data.min(), x2Data.max()
		
		trace = go.Heatmap(x=x1Data, y=x2Data, z=yData)
		data = [trace]

		layout = Layout(
			title='%s -  C = %s, G = %s, D = %s ' %(title, C, G, D),
			scene=Scene(
				zaxis=ZAxis(title = 'Likl',range=[x1Data.min(),x1Data.max()] ,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
				xaxis=XAxis(title = 'Rain', range=[x2Data.min(),x2Data.max()],gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
				yaxis=YAxis(title = 'Erod', range=[yData.min(),yData.max()] ,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
				bgcolor="rgb(244, 244, 248)"
			)
		)

		fig = Figure(data=data, layout=layout)
		graph = plotly.offline.plot(fig, auto_open=False, output_type='file', filename='%s/plots/_grid_%s_%s_%s_%s.html' %(self.filename, title, C, G, D), validate=False)
		return

	def surrogate_model(self):
		"""
		Main entry point for running badlands model with different forcing conditions.
		"""
		file = np.loadtxt("%s/exp_data.txt" % self.filename, delimiter = '\t')
		data = file[:,0:1].reshape(file.shape[0], 1)
		# data = data[:,0].reshape(data.shape[0],1)
		# y = np.asarray(file[:,2])
		y = np.asarray(file[:,3]).reshape(file.shape[0],1)
		y = normalize(y, axis = 0, norm = 'l2') # 'l1, l2, max'
		y = y.ravel()
		# y = y*(-1)

		X = np.asarray(data)
		
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
		print 'Y\n', y 
		print y.shape
		print X_train.shape
		print y_train.shape
		print X_test.shape
		print y_test.shape

		# First run
		plt.figure(0)
		kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
		    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
		gp = GaussianProcessRegressor(kernel=kernel,
		                              alpha=0.0).fit(X_train, y_train)
		# X_test = np.linspace(0, 5, 100)
		y_mean, y_cov = gp.predict(X_test, return_cov=True)
		plt.plot(X_test, y_mean, 'k', lw=3, zorder=9)
		# plt.fill_between(X_test, y_mean - np.sqrt(np.diag(y_cov)), y_mean + np.sqrt(np.diag(y_cov)),alpha=0.5, color='k')
		plt.plot(X_test, 0.5*np.sin(3*X_test), 'r', lw=3, zorder=9)
		plt.scatter(X_train[:, 0], y_train, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
		# plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"% (kernel, gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)))
		plt.tight_layout()

		# Second run
		plt.figure(1)
		kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
		    + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
		gp = GaussianProcessRegressor(kernel=kernel,
		                              alpha=0.0).fit(X_train, y_train)
		# X_test = np.linspace(0, 5, 100)
		y_mean, y_cov = gp.predict(X_test, return_cov=True)
		plt.plot(X_test, y_mean, 'k', lw=3, zorder=9)
		# plt.fill_between(X_test, y_mean - np.sqrt(np.diag(y_cov)),y_mean + np.sqrt(np.diag(y_cov)),alpha=0.5, color='k')
		plt.plot(X_test, 0.5*np.sin(3*X_test), 'r', lw=3, zorder=9)
		plt.scatter(X_train[:, 0], y_train, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
		# plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s" %(kernel, gp.kernel_,gp.log_marginal_likelihood(gp.kernel_.theta)))
		plt.tight_layout()

		# Plot LML landscape
		plt.figure(2)
		theta0 = np.logspace(-2, 3, 49)
		theta1 = np.logspace(-2, 0, 50)
		Theta0, Theta1 = np.meshgrid(theta0, theta1)
		LML = [[gp.log_marginal_likelihood(np.log([0.36, Theta0[i, j], Theta1[i, j]]))
		        for i in range(Theta0.shape[0])] for j in range(Theta0.shape[1])]
		LML = np.array(LML).T

		vmin, vmax = (-LML).min(), (-LML).max()
		vmax = 50
		level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 50), decimals=1)
		plt.contour(Theta0, Theta1, -LML,
		            levels=level, norm=LogNorm(vmin=vmin, vmax=vmax))
		plt.colorbar()
		plt.xscale("log")
		plt.yscale("log")
		plt.xlabel("Length-scale")
		plt.ylabel("Noise-level")
		plt.title("Log-marginal-likelihood")
		plt.tight_layout()

		plt.show()
		plt.savefig('%s/plots/_%s_%s_%s.png'% (self.filename ,C,G,D), bbox_inches='tight', dpi=300, transparent=False)

		self.viewGrid(X_test[:,0], X_test[:,1], y_rbf, C,G,D,self.filename, 'SVR - rbf')
		self.viewGrid(X_test[:,0], X_test[:,1], y_poly, C,G,D,self.filename, 'SVR - poly')
		self.viewGrid(X_test[:,0], X_test[:,1], y_lin, C,G,D,self.filename, 'SVR- linear')

def main():
	"""
		
	"""
	random.seed(time.time())
	run_nb = 0
	directory = 'Examples/crater'

	if os.path.exists('%s/c_liklsurf_s_true' % (directory)):
		filename = ('%s/c_liklsurf_s_true' % (directory))

	run_nb_str = 'c_liklsurf_s_true' + str(run_nb)

	pt_sur_gp = pt_surrogate_GP(filename, run_nb_str)
	pt_sur_gp.surrogate_model()

	print '\nsuccessfully sampled\nFinished simulations'

if __name__ == "__main__": main()