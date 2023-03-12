import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
save_path = os.path.join(root_dir, 'figures')

TOL = 1e-14

def visualize_heatmap(x, y, heat_list, heat_pre_list, epoch):
    #Creates a matplotlib of the heatmat for ease of understanding
    plt.figure(figsize=(18,25))
    num = len(heat_list)
    for i in range(num):
        plt.subplot(num, 3, i*3+1)
        plt.contourf(x, y, heat_list[i], levels=50, cmap=matplotlib.cm.coolwarm) # makes a contour map
        plt.colorbar() #legend
        plt.title('True') #This subplot is for the true value
        plt.subplot(num, 3, i*3+2)
        plt.contourf(x, y, heat_pre_list[i], levels=50, cmap=matplotlib.cm.coolwarm)
        plt.colorbar()
        plt.title('Prediction') #This subplot is for the predicted heat
        plt.subplot(num, 3, i*3+3)
        plt.contourf(x, y, heat_pre_list[i] - heat_list[i], levels=50, cmap=matplotlib.cm.coolwarm)
        plt.colorbar()
        plt.title('Error') #This subplot is for the "Error" - the difference of the two which correlates to the error
    plt.savefig(save_path + '/epoch' + str(epoch) + '_pre.png', bbox_inches='tight', pad_inches=0)
    plt.close()