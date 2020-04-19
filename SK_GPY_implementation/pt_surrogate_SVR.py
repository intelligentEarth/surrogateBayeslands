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
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
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

class pt_surrogate_SVR():
	"""
		
	"""
	def __init__(self, filename, run_nb):
		self.filename = filename
		self.run_nb = run_nb
	
	def viewHeatmap(self, x1Data, x2Data, yData, C, G, D, filename, title):
		print x1Data.min(), x1Data.max()
		print x2Data.min(), x2Data.max()
		# print 'yrbfData', yrbfData
		# yrbfsurf = Surface( x=x1Data, y=x2Data, z=yrbfData, colorscale='YIGnBu')
		# # ylinsurf = Surface( x=x1Data, y=x2Data, z=ylinData, colorscale='YIGnBu')
		# # ypolysurf = Surface( x=x1Data, y=x2Data, z=ypolyData, colorscale='YIGnBu')
		# # data = Data([yrbfsurf, ylinsurf, ypolysurf])
		# data = Data([Surface( x=x1Data, y=x2Data, z=yrbfData, colorscale='YIGnBu')])
		
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
		data = file[:,0:2].reshape(file.shape[0], 2)
		# data = data[:,0].reshape(data.shape[0],1)
		# y = np.asarray(file[:,2])
		y = np.asarray(file[:,3]).reshape(file.shape[0],1)
		y = normalize(y, axis = 0, norm = 'max') # 'l1, l2, max'
		y = y.ravel()
		# y = y*(-1)

		X = np.asarray(data)

		C = 5000
		G = 0.1
		D = 2

		self.viewHeatmap(X[:,0], X[:,1], y,C,G,D, self.filename, 'Likelihood Data')

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
		print 'Y\n', y 
		print y.shape
		print X_train.shape
		print y_train.shape
		print X_test.shape
		print y_test.shape


		svr_rbf = SVR(kernel='rbf', C=C, gamma=G)
		svr_lin = SVR(kernel='linear', C=C)
		svr_poly = SVR(kernel='poly', C=C, degree=D)
		y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)
		y_lin = svr_lin.fit(X_train, y_train).predict(X_test)
		y_poly = svr_poly.fit(X_train, y_train).predict(X_test)

		print 'RBF'
		# The coefficients
		# print 'RBF Coefficients: \n', y_rbf.coef_
		# The mean squared error
		print "Mean squared error: %.2f"% mean_squared_error(y_test, y_rbf)
		# Explained variance score: 1 is perfect prediction
		print 'Variance score: %.2f' % r2_score(y_test, y_rbf)

		print 'Linear'
		# The coefficients
		# print 'Linear Coefficients: \n', y_lin.coef_
		# The mean squared error
		print "Mean squared error: %.2f"% mean_squared_error(y_test, y_lin)
		# Explained variance score: 1 is perfect prediction
		print 'Variance score: %.2f' % r2_score(y_test, y_lin)

		print 'Poly'
		# The coefficients
		# print 'Poly Coefficients: \n', y_poly.coef_
		# The mean squared error
		print "Mean squared error: %.2f"% mean_squared_error(y_test, y_poly)
		# Explained variance score: 1 is perfect prediction
		print 'Variance score: %.2f' % r2_score(y_test, y_poly)

		lw = 1
		plt.scatter(X_test[:,0], y_test, color='darkorange', label='data')
		plt.plot(X_test[:,0], y_rbf, color='navy', lw=lw, label='RBF model')
		plt.plot(X_test[:,0], y_lin, color='c', lw=lw, label='Linear model')
		plt.plot(X_test[:,0], y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
		plt.xlabel('data')
		plt.ylabel('target')
		plt.title('Support Vector Regression')
		plt.legend()
		# plt.show()
		print '%s/_%s_%s_%s.png'% (self.filename ,C,G,D)
		plt.savefig('%s/plots/_%s_%s_%s.png'% (self.filename ,C,G,D), bbox_inches='tight', dpi=300, transparent=False)

		self.viewHeatmap(X_test[:,0], X_test[:,1], y_rbf, C,G,D,self.filename, 'SVR - rbf')
		self.viewHeatmap(X_test[:,0], X_test[:,1], y_poly, C,G,D,self.filename, 'SVR - poly')
		self.viewHeatmap(X_test[:,0], X_test[:,1], y_lin, C,G,D,self.filename, 'SVR- linear')

def main():
	"""
		
	"""
	random.seed(time.time())
	run_nb = 0
	directory = 'Examples/crater'

	if os.path.exists('%s/c_liklsurf_s_true' % (directory)):
		filename = ('%s/c_liklsurf_s_true' % (directory))

	run_nb_str = 'c_liklsurf_s_true' + str(run_nb)

	pt_sur_svr = pt_surrogate_SVR(filename, run_nb_str)
	pt_sur_svr.surrogate_model()

	print '\nsuccessfully sampled\nFinished simulations'

if __name__ == "__main__": main()