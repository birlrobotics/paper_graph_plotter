import glob
import os
import coloredlogs, logging
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12})
import pandas as pd
import ipdb

coloredlogs.install()

if __name__ == '__main__':
    logger = logging.getLogger()
    dir_of_this_script = os.path.dirname(os.path.realpath(__file__))
    for csv in glob.glob(os.path.join(dir_of_this_script, "whole_success_exp", "*", "*csv")):
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
        df.plot(ax=ax, x=df.keys()[0], y=df.keys()[1:7],  colormap='jet',legend=False)
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
        ax.set_ylabel("Force/Torque")
        output_dir = os.path.join(dir_of_this_script, 'plots')        
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        f.savefig(os.path.join(output_dir, "%s.png"%relpath[-38:]),dpi=300)
        plt.show()
        
