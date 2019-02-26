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

def plot_in_2d():
    logger = logging.getLogger()
    dir_of_this_script = os.path.dirname(os.path.realpath(__file__))
    for csv in glob.glob(os.path.join(dir_of_this_script, "whole_experiment_with_endpoint_pose", "*", "*csv")):
        f,ax = plt.subplots(nrows=1, ncols=1)
        ax.set_title("Multimodal signals of whole non-anomalous experiment")
        relpath = os.path.relpath(csv, os.path.join(dir_of_this_script))
        logger.info(relpath) 
        df = pd.read_csv(csv)
        s_t = df.iloc[0, 0]
        df.iloc[:, 0] = df.iloc[:, 0]-s_t

        tag_ranges = pickle.load(open(os.path.join(os.path.dirname(csv), 'tag_ranges.pkl'), 'rb'))
        tag_stime = []
        for tag, (st, et) in tag_ranges:
            tag_stime.append((tag, st.to_sec()-s_t, et.to_sec()-s_t))

        csv_name = os.path.basename(relpath)
        df.plot(ax=ax, x=df.keys()[0], y=df.keys()[-3:],  colormap='jet',marker='.', legend=True)
        xmin,xmax = ax.get_xlim()
        ymin,ymax = ax.get_ylim()

        for tag, start, end in tag_stime:
            if int(tag) != 0:
                if int(tag) < 0:
                    print ('Ignore the anomaly tag < 0')
                    continue
                    c = 'red'
                else:
                    c = 'green'
                ax.axvline(start, c=c)
                ax.text(start, ymax-0.05*(ymax-ymin), 'skill %s'%tag, color=c, rotation=-90)
            else:
                print ('Ignore the useless tag=0')
                continue
                ax.axvline(start, c = 'gray')
        ax.set_xlabel("Time(secs)")
        ax.set_ylabel("Scaled Magnitude")
        output_dir = os.path.join(dir_of_this_script, 'plots')        
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        f.savefig(os.path.join(output_dir, "%s.png"%relpath[-38:]),dpi=300)
    plt.show()

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
#              9:'y'
                  }
    for i, csv in enumerate(glob.glob(os.path.join(dir_of_this_script, "whole_experiment_with_endpoint_pose", "*", "*csv"))):
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
                if int(tag) < 0 or int(tag) == 9:
                    print ('Ignore the anomaly tag < 0 or tag == 9')
                    continue
                else:
                    tag_df = df.loc[(df['Unnamed: 0'] >= start) & (df['Unnamed: 0'] <= end)]
                    ax.plot(tag_df[df.keys()[-3]],tag_df[df.keys()[-2]],
                            tag_df[df.keys()[-1]], color=colors[tag], label='Skill ' + str(tag))
                                        
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
    fig.savefig(os.path.join(output_dir, "dmp.png"),dpi=300)
    plt.show()
        
if __name__ == '__main__':
    plot_in_3d()
