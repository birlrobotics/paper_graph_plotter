import glob
import os
import coloredlogs, logging
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
import pandas as pd
import ipdb

coloredlogs.install()

def plot_in_3d():
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    from collections import OrderedDict
    logger = logging.getLogger()
    dir_of_this_script = os.path.dirname(os.path.realpath(__file__))
    fig = plt.figure()
    ax = fig.gca(projection='3d', adjustable='box')
    colors = {3:'r',
              4:'g',
              5:'b',
              7:'c',
              8:'m',
              9:'y',
              -1:'grey', # anomalies
              -2:'gold', # reactment
              -3:'black', # adaptation
              1000:'indigo', # recovery behavior
              1001:'teal',
              1002:'k',}
    position = ['baxter_enpoint_pose.pose.position.x','baxter_enpoint_pose.pose.position.y','baxter_enpoint_pose.pose.position.z']        
    for i, csv in enumerate(glob.glob(os.path.join(dir_of_this_script, "recovery_success_with_endpoint_pose", "*", "*csv"))):
        relpath = os.path.relpath(csv, os.path.join(dir_of_this_script))
        logger.info(relpath)
        df = pd.read_csv(csv)
        s_t = df.iloc[0, 0]
        df.iloc[:, 0] = df.iloc[:, 0]-s_t
        
        tag_ranges = pickle.load(open(os.path.join(os.path.dirname(csv), 'tag_ranges.pkl'), 'rb'))
        tag_stime = []
        for tag, (st, et) in tag_ranges:
            tag_stime.append((tag, st.to_sec()-s_t, et.to_sec()-s_t))
        for tag, start, end in tag_stime:
            if int(tag) != 0:
                if int(tag) < -3:
                    print ('Ignore the anomaly tag < 0')
                    continue
                else:
                    tag_df = df.loc[(df['Unnamed: 0'] >= start) & (df['Unnamed: 0'] <= end)]
                    ax.plot(tag_df[position[0]],
                            tag_df[position[1]],
                            tag_df[position[2]], color=colors[tag], label='Skill ' + str(tag))
                                        
            else:
                print ('Ignore the useless tag=0')
                continue
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
    fig.savefig(os.path.join(output_dir, "recovey_success_dmp.png"),dpi=300)
    plt.show()
        
if __name__ == '__main__':
    plot_in_3d()
