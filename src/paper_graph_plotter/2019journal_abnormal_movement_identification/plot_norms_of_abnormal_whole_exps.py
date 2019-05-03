import glob
import os
import coloredlogs, logging
import pickle
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import ipdb
import numpy as np

matplotlib.rcParams.update({'font.size': 12})
plt.rcParams["font.family"] = "Time New Roman"


coloredlogs.install()

if __name__ == '__main__':
    logger = logging.getLogger()
    dir_of_this_script = os.path.dirname(os.path.realpath(__file__))
    norms = [r'$n_f$', r'$n_m$', r'$n_l$', r'$n_a$', r'$s_l$', r'$s_r$',]
    f,axarr = plt.subplots(nrows=len(norms)/2, ncols=2, sharex=True, figsize=(10, 5))
    axarr = np.atleast_1d(axarr).flatten().tolist()
    f.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.1)
    FLAG = True
    colors = {3:'r',
              4:'g',
              5:'b',
              7:'c',
              8:'m',
              9:'y',
              0:'grey'}
    for i, csv in enumerate(glob.glob(os.path.join(dir_of_this_script, "whole_experiment_with_anomalies", "*", "*csv"))):
        print i
        print csv
        print 
        relpath = os.path.relpath(csv, os.path.join(dir_of_this_script))
        logger.info(relpath) 
        df = pd.read_csv(csv)
        s_t = df.iloc[0, 0]
        df.iloc[:, 0] = df.iloc[:, 0]-s_t

        tag_ranges = pickle.load(open(os.path.join(os.path.dirname(csv), 'tag_ranges.pkl'), 'rb'))
        tag_stime = []
        for tag, (st, et) in tag_ranges:
            tag_stime.append((tag, st.to_sec()-s_t, et.to_sec()-s_t))
        tag_list = []
        csv_name = os.path.basename(relpath)
        keys = df.keys()
        for j, norm  in enumerate(norms):
            ax = axarr[j]
            if FLAG: ax.set_title(norms[j])
            if FLAG: ax.set_ylabel("Value")
            for tag, start, end in tag_stime:
                if tag not in colors.keys(): continue
                tag_df = df.loc[(df['Unnamed: 0'] >= start) & (df['Unnamed: 0'] <= end)]
                ax.plot(tag_df[keys[0]].values, tag_df[keys[j+1]].values, c=colors[tag],lw=1.5)
                tag_list.append(tag)
        FLAG = False
        print np.unique(np.array(tag_list))
    axarr[len(norms)-1].set_xlabel("Time(secs)")
    axarr[len(norms)-2].set_xlabel("Time(secs)")    
    output_dir = os.path.join(dir_of_this_script, 'plots')        
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    f.savefig(os.path.join(output_dir, "multi_whole_abnormal_exps_norms.png"),format='png', dpi=300)
    f.savefig(os.path.join(output_dir, "multi_whole_abnormal_exps_norms.eps"),format='eps', dpi=300)    
    plt.show()
        
