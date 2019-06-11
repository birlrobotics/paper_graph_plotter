import glob
import os
import coloredlogs, logging
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12})
plt.rcParams["font.family"] = "Time New Roman"
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from collections import OrderedDict
import pandas as pd
import ipdb

coloredlogs.install()

if __name__ == '__main__':
    logger = logging.getLogger()
    dir_of_this_script = os.path.dirname(os.path.realpath(__file__))
    
    data_path = '/home/birl-spai-ubuntu14/baxter_ws/src/SPAI/smach_based_introspection_framework/jim_folder_have_tag_range/anomaly_detection_feature_selection_folder/No.0 filtering scheme/whole_experiment'
    
    fig = plt.figure()
    ax = fig.gca(projection='3d', adjustable='box')
    
    for i, csv in enumerate(glob.glob(os.path.join(data_path, "*", "*csv"))):
        print i
        print csv
        print 
#        ax.set_title("Multimodal signals of norminal execution #%s" %(i+1))
#       
        df = pd.read_csv(csv)
        s_t = df.iloc[0, 0]
        df.iloc[:, 0] = df.iloc[:, 0]-s_t
        select_list = ['baxter_enpoint_pose.pose.position.x','baxter_enpoint_pose.pose.position.y','baxter_enpoint_pose.pose.position.z']
        values = df[select_list].values
        ax.plot(values[:,0],values[:,1],values[:,2], label = "position")
        ax.scatter(values[0,0],values[0,1],values[0,2], c = "gray", label = "start")
        ax.scatter(values[-1,0],values[-1,1],values[-1,2], c = 'blue' , label = "end")        
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    # Get rid of colored axes planes
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    # Bonus: To get rid of the grid as well:
    ax.grid(False)
    plt.axis('off')
    
    output_dir = os.path.join(dir_of_this_script, 'plots')        
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if 'unsuccessful' in data_path.lower():
        file_name = "unsuccess.png"
    else:
        file_name = "success.png"
    fig.savefig(os.path.join(output_dir, file_name),dpi=300)
    plt.show()
