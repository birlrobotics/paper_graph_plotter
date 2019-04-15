import glob
import os
import coloredlogs, logging
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import ipdb
import numpy as np


coloredlogs.install()
plt.rcParams.update({'font.size': 12})
plt.rcParams["font.family"] = "Time New Roman"


if __name__ == '__main__':
    logger = logging.getLogger("plot_anomalous_signals.main")
    dir_of_this_script = os.path.dirname(os.path.realpath(__file__))
    nfiles = len(glob.glob(os.path.join(dir_of_this_script, "whole_exp_with_one_anomaly", "*", "*csv")))
    f,axarr = plt.subplots(nrows=4, ncols=2, sharex =True, figsize=(16,10))
    axarr = np.atleast_1d(axarr).flatten().tolist()
    #f.tight_layout()    
    for i, csv in enumerate(glob.glob(os.path.join(dir_of_this_script, "whole_exp_with_one_anomaly", "*", "*csv"))):
        ax = axarr[i]
        relpath = os.path.relpath(csv, os.path.join(dir_of_this_script))
        logger.info(relpath) 

        df = pd.read_csv(csv)
        df = df.ix[0:800]
        s_t = df.iloc[0, 0]
        df.iloc[:, 0] = df.iloc[:, 0]-s_t
        
        anomaly_signals = pickle.load(open(os.path.join(os.path.dirname(csv), "anomaly_signals.pkl"), 'rb'))
        tag_ranges = pickle.load(open(os.path.join(os.path.dirname(csv), 'tag_ranges.pkl'), 'rb'))
        SUCCESS = False
        if len(anomaly_signals) < 1:
            SUCCESS = True
            anomaly_signals = [('success', tag_ranges[0][1][0])]       
        anomaly_type = anomaly_signals[0][0]
        anomaly_gentime = anomaly_signals[0][1]

        print i
        print csv
        print anomaly_type
        print
        ax.set_title("Multimodal signals of #%s#" %anomaly_type)

        tag_stime = []
        for tag, (st, et) in tag_ranges:
            tag_stime.append((tag, st.to_sec()-s_t, et.to_sec()-s_t))

        anomaly_time = anomaly_gentime.to_sec()-s_t
        csv_name = os.path.basename(relpath)
        df.plot(ax=ax, x=df.keys()[0], y=df.keys()[1::],  colormap='jet', legend=False)
        xmin,xmax = ax.get_xlim()
        ymin,ymax = ax.get_ylim()

        if not SUCCESS:        
            ax.axvline(anomaly_time, c='black', ls = '--')
            ax.axvspan(anomaly_time - 2, anomaly_time + 2, ymin=0.0, ymax=1.0, facecolor='red', alpha=0.6)
        
        skill = 0
        for tag, start, end in tag_stime:
            if int(tag) != 0:
                if int(tag) < 0:
                    print ('Ignore the anomaly tag < 0')
                    continue
                    c = 'red'
                else:
                    c = 'green'
                ax.axvline(start, c=c)
                skill += 1
                ax.text(start, ymax-0.05*(ymax-ymin), 'Skill %s'%skill, color=c, rotation=-90)
            else:
                print ('Ignore the useless tag=0')
                continue
                ax.axvline(start, c = 'gray')
        
        ax.text(anomaly_time, ymax-0.05*(ymax-ymin), anomaly_type, color='red', rotation=0)
        ax.set_xlabel("Time(secs)")
        ax.set_ylabel("Scaled Value")
    output_dir = os.path.join(dir_of_this_script, 'plots')        
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    f.align_labels()
    plt.subplots_adjust(hspace=0.35)
    f.savefig(os.path.join(output_dir, "%s.png"% ('multi_whole_exps_with_anomaly')),dpi=300)
    plt.show()
    plt.close(f)
        
