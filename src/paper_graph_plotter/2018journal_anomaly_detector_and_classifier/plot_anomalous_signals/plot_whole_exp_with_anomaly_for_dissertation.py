import glob
import os
import coloredlogs, logging
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import ipdb

coloredlogs.install()

if __name__ == '__main__':
    logger = logging.getLogger("plot_anomalous_signals.main")
    dir_of_this_script = os.path.dirname(os.path.realpath(__file__))
    for i, csv in enumerate(glob.glob(os.path.join(dir_of_this_script, "whole_exp_with_one_anomaly", "*", "*csv"))):
        f,ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
        fontsize=12
        #ax.set_title("Multimodal Signals of Tool Collision", fontsize=fontsize)
        relpath = os.path.relpath(csv, os.path.join(dir_of_this_script))
        logger.info(relpath) 

        df = pd.read_csv(csv)
        s_t = df.iloc[0, 0]
        df.iloc[:, 0] = df.iloc[:, 0]-s_t
        
        anomaly_signals = pickle.load(open(os.path.join(os.path.dirname(csv), "anomaly_signals.pkl"), 'rb'))
        anomaly_type = anomaly_signals[0][0]
        anomaly_gentime = anomaly_signals[0][1] 

        tag_ranges = pickle.load(open(os.path.join(os.path.dirname(csv), 'tag_ranges.pkl'), 'rb'))
        tag_stime = []
        for tag, (st, et) in tag_ranges:
            tag_stime.append((tag, st.to_sec()-s_t, et.to_sec()-s_t))

        anomaly_time = anomaly_gentime.to_sec()-s_t
        csv_name = os.path.basename(relpath)
        df.plot(ax=ax, x=df.keys()[0], y=df.keys()[1::],  colormap='jet',legend=False)
        xmin,xmax = ax.get_xlim()
        ymin,ymax = ax.get_ylim()
        
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
                ax.text(start, ymax-0.05*(ymax-ymin), 'skill %s'%skill, color=c, fontsize=fontsize, rotation=-90)
            else:
                print ('Ignore the useless tag=0')
                continue
                ax.axvline(start, c = 'gray')
        
        #ax.text(anomaly_time, ymax-0.05*(ymax-ymin), anomaly_type, color='red', fontsize=fontsize, rotation=-90)
        ax.set_xlabel("Time(secs)", fontsize=fontsize)
        ax.set_ylabel("Scaled Value", fontsize=fontsize)
        output_dir = os.path.join(dir_of_this_script, 'plots')        
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        f.savefig(os.path.join(output_dir, "%s_of_%s.jpg"% ('whole_exp_with_anomaly',i)),dpi=300)
        plt.show()
        plt.close(f)
        
