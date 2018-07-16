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
    f,axarr = plt.subplots(nrows=2, ncols=1, sharex=True)
    axarr[0].set_title("Multimodal Signals during Wall Collision")
    f.subplots_adjust(hspace=0.10)
    nplot = 0
    for csv in glob.glob(os.path.join(dir_of_this_script, "wall_collison_csv_files", "*", "*csv")):
        relpath = os.path.relpath(csv, os.path.join(dir_of_this_script))
        logger.info(relpath) 
	
        anomaly_gentime = pickle.load(open(os.path.join(os.path.dirname(csv), "anomaly_time.pkl"), 'rb'))
        

        df = pd.read_csv(csv)
        s_t = df.iloc[0, 0]
        df.iloc[:, 0] = df.iloc[:, 0]-s_t
        anomaly_time = anomaly_gentime.to_sec()-s_t

        csv_name = os.path.basename(relpath)
        ax=axarr[nplot]
        df.plot(ax=ax, x=df.keys()[0], y=df.keys()[1::],  colormap='jet',marker='.', legend=False)
        nplot += 1
        ax.axvline(anomaly_time, c='red', ls = '--')

        xmin,xmax = ax.get_xlim()
        ymin,ymax = ax.get_ylim()
        ax.text(anomaly_time, ymax-0.25*(ymax-ymin), 'wall collision', color='red', rotation=-90)

        ax.set_xlabel("Time(secs)", fontsize=10)
        ax.set_ylabel("Scaled Magnitude", fontsize=10)
    output_dir = os.path.join(dir_of_this_script, 'plots')        
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    f.savefig(os.path.join(output_dir, "%s.eps"%  'Comaprision_of_wall_collision'),dpi=300)
    plt.show()
    plt.close(f)
        
