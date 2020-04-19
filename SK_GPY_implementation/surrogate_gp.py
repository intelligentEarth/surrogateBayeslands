import numpy as np
import time
from matplotlib import pyplot as plt
import plotly
import plotly.plotly as py
import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.graph_objs import *
import plotly.graph_objs as go
from plotly.offline.offline import _plot_html
import GPy
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import normalize
np.random.seed(1)
plotly.offline.init_notebook_mode()
GPy.plotting.change_plotting_library('plotly')
plotly.tools.set_credentials_file(username='dazam92', api_key='cZseSTjG7EBkTasOs8Op')

def SurrogateHeatmap(filename, x1Data, x2Data, yData, y_min, y_max, title):
	trace = go.Heatmap(x=x1Data, y=x2Data, z=yData, zmin = y_min, zmax = y_max)
	data = [trace]
	layout = Layout(
		title='%s ' %(title),
		scene=Scene(
			zaxis=ZAxis(title = 'Likl',range=[-1,0] ,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
			xaxis=XAxis(title = 'Rain', range=[x2Data.min(),x2Data.max()],gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
			yaxis=YAxis(title = 'Erod', range=[x2Data.min(),x2Data.max()] ,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
			bgcolor="rgb(244, 244, 248)"
		)
	)
	fig = Figure(data=data, layout=layout)
	plotly.offline.plot(fig, auto_open=False, output_type='file', filename='%s/%s.html' %(filename, title), validate=False)
	return

def main():
	
	choice = input("1- Crater\n2- Etopo\n")
	directory = ""
	if choice == 1:
		problem = "crater"
		directory = "Examples/crater/c_liklsurf_s_true/"
		file = np.loadtxt("%sexp_data.txt" %directory, delimiter = '\t')
	elif choice == 2:
		problem = "etopo"
		directory = "Examples/etopo/e_liklsurf_s_true/"
		file = np.loadtxt("%sexp_data.txt" %directory, delimiter = '\t')
		
	data = file[:,0:2].reshape(file.shape[0], 2)
	y = np.asarray(file[:,2]).reshape(file.shape[0],1)
	y = y.ravel()
	y = y/(-1*y.min())
	y = y.reshape(file.shape[0], 1)
	y_min = y.min()
	y_max = y.max()
	
	X = np.asarray(data)

	SurrogateHeatmap(directory, X[:,0], X[:,1], y.ravel(), y_min, y_max, 'GPY- data')

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	print 'y.shape', y.shape
	print 'X.shape', X.shape
	print 'X_train.shape', X_train.shape
	print 'y_train.shape', y_train.shape 
	print 'X_test.shape', y_train.shape
	print 'y_test.shape', y_test.shape

	start = time.time()
	# define kernel
	ker = GPy.kern.Matern52(input_dim = 2, lengthscale = 1., ARD=True) + GPy.kern.White(2)
	# kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
	# ker = GPy.kern.RBF(2,ARD=True) + GPy.kern.White(2)

	# create simple GP model
	gp = GPy.models.GPRegression(X_train,y_train,ker)

	# optimize and plot
	# gp.optimize(messages=True, max_f_eval = 1000)
	gp.optimize_restarts(messages = True, num_restarts = 2)

	fig = gp.plot()

	y_pred = gp.predict(X_test)[0]

	print "Mean squared error: %.2f"% mean_squared_error(y_test.ravel(), y_pred.ravel())
	print 'Variance score: %.2f' % r2_score(y_test.ravel(), y_pred.ravel())
	print '\nTime for GP training ', time.time() - start, 'sec \n'

	SurrogateHeatmap(directory, X_test[:,0], X_test[:,1], y_test.ravel(), y_min, y_max, 'GPY - test_data')
	SurrogateHeatmap(directory, X_test[:,0], X_test[:,1], y_pred.ravel(), y_min, y_max,'GPY - GP_pred')
	
	# display(GPy.plotting.show(fig, filename='basic_gp_regression_notebook_2d'))
	# plotly.offline.plot(gp, auto_open = False, output_type = 'file', filename = 'Examples/crater/c_liklsurf_s_true/GPy/%s.html' %(problem), validate=False)
	# plotly.offline.plot(fig, auto_open = False, output_type = 'file', filename = 'Examples/crater/c_liklsurf_s_true/GPy/%s.html' %(problem), validate=False)
	# display(gp)

	print 'Optimized parameters', gp.param_array, '\n'
	
	start = time.time()
	np.save('gp_params.npy', gp.param_array)
	np.save('gp_y.npy',y_train)
	np.save('gp_X.npy',X_train)
	# np.save('gp_L.npy', gp.L_)

	# Load model
	y_load = np.load('gp_y.npy')
	X_load = np.load('gp_X.npy')
	# L_load = np.load('gp_L.npy')

	print '\nTime saving and loading Model ', time.time() - start, 'sec \n'

	start = time.time()
	gp_load = GPy.models.GPRegression(X_load, y_load, initialize=False,kernel=ker) # Kernel is problematic here
	gp_load.update_model(False)
	gp_load.initialize_parameter()
	gp_load[:] = np.load('gp_params.npy')
	gp_load.update_model(True)
	# gp_load.L_ = L_load
	print '\nTime for GP updating ', time.time() - start, ' sec \n'
	
	y_pred_gpload = gp_load.predict(X_test)[0]
	print "Mean squared error for loaded GP: %.2f"% mean_squared_error(y_test.ravel(), y_pred_gpload.ravel())
	print 'Variance score for loaded GP: %.2f' % r2_score(y_test.ravel(), y_pred_gpload.ravel())
	print '\nTime for GP loaded prediction ', time.time() - start, 'sec \n'
	SurrogateHeatmap(directory, X_test[:,0], X_test[:,1], y_pred_gpload.ravel(), y_min, y_max,'GPY - GP_load_pred')
	
if __name__ == "__main__": main()