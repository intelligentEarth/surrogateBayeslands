import numpy as np
batch_ratios = np.linspace(0.1, 0.5, 5)
dropout_values = np.linspace(0.0, 0.5, 5)
problems = ['etopo', 'crater', 'mountain']
mean_file = open('mean.txt', 'w')
for dropout in dropout_values[1:2]:
    for problem in problems:
        for ratio in batch_ratios:
            ratio = np.round(ratio, decimals=2)
            dropout = np.round(dropout, decimals=2)
            mse_list = []
            with open('Results/results_10000_adam_no_trf/results_'+str(problem)+'_10000_'+str(ratio)+'_'+str(dropout)+'/results.txt', 'r') as file:
                lines = file.readlines()
                for line in lines[:-1]:
                    # print(line.split())
                    mse = np.round(float(line.split()[-1]), 4)
                    mse_list.append(mse)
                print(lines[-1].split()[-1])
            mean_mse = np.round(np.array(mse_list).mean(), 4)
            # print(problem+'_'+str(ratio)+'_'+str(dropout)+' ', mean_mse)
            mean_file.write(str(mean_mse)+'\n')
mean_file.close()
