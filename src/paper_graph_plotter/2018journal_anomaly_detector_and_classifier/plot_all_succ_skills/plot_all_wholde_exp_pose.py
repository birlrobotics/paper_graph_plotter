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
import pandas as pd
from collections import OrderedDict
import ipdb
coloredlogs.install()

if __name__ == '__main__':
    logger = logging.getLogger()
    dir_of_this_script = os.path.dirname(os.path.realpath(__file__))
    data_path = "/home/birl-spai-ubuntu14/baxter_ws/src/SPAI/smach_based_introspection_framework/jim_folder_have_tag_range/anomaly_detection_feature_selection_folder/No.0 filtering scheme/whole_experiment"
    
    fig = plt.figure()
    ax = fig.gca(projection='3d', adjustable='box')
    colors = {3:'r',
              4:'g',
              5:'b',
              7:'c',
              8:'m',
              9:'y'
                  }
    for i, csv in enumerate(glob.glob(os.path.join(data_path, "*", "*csv"))):
        df = pd.read_csv(csv)
        s_t = df.iloc[0, 0]
        df.iloc[:, 0] = df.iloc[:, 0]-s_t

        tag_ranges = pickle.load(open(os.path.join(os.path.dirname(csv), 'tag_ranges.pkl'), 'rb'))
        tag_stime = []
        for tag, (st, et) in tag_ranges:
            tag_stime.append((tag, st.to_sec()-s_t, et.to_sec()-s_t))
        

        for tag, start, end in tag_stime:
            if int(tag) != 0:
                if int(tag) < 0 or int(tag) >= 1000:
                    print ('Ignore the anomaly tag < 0 or tag >= 1000')
                    continue
                else:
                    tag_df = df.loc[(df['Unnamed: 0'] >= start) & (df['Unnamed: 0'] <= end)]
                    select_list = ['baxter_enpoint_pose.pose.position.x',
                                   'baxter_enpoint_pose.pose.position.y',
                                   'baxter_enpoint_pose.pose.position.z']
                    values = tag_df[select_list].values
                    # ax.plot(values[:,0],values[:,1],values[:,2], color=colors[tag], label='Skill ' + str(tag), alpha=0.5)
                    try:
                        ax.scatter(values[0,0],values[0,1],values[0,2], c = colors[tag], label ='Skill ' + str(tag), alpha=0.5)
                        #ax.scatter(values[-1,0],values[-1,1],values[-1,2], c = 'blue' , label = "end")        
                    except Exception as e:
                        logger.error("error encountered ")
                        continue
                    
        if i > 300: break
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
       
    output_dir = os.path.join(dir_of_this_script, 'plots')        
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    fig.savefig(os.path.join(output_dir, "multi_whole_exps.png"),format='png', dpi=300)
    plt.show()
        

        
