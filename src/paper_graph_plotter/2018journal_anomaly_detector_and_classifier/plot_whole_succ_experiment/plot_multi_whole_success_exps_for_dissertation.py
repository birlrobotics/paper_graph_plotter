import glob
import os
import coloredlogs, logging
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12})
plt.rcParams["font.family"] = "Time New Roman"

import pandas as pd
import ipdb

coloredlogs.install()

if __name__ == '__main__':
    logger = logging.getLogger()
    dir_of_this_script = os.path.dirname(os.path.realpath(__file__))
    nfiles = len(glob.glob(os.path.join(dir_of_this_script, "multi_whole_success_exps", "*", "*csv")))
    f,axarr = plt.subplots(nrows=nfiles, ncols=1, sharex=True, figsize=(8, 7))
    f.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    for i, csv in enumerate(glob.glob(os.path.join(dir_of_this_script, "multi_whole_success_exps", "*", "*csv"))):
        print i
        print csv
        print 
        ax = axarr[i]
        ax.set_title("Multimodal signals of norminal execution #%s" %(i+1))
        ax.set_xlim([0, 60])
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
        df.plot(ax=ax, x=df.keys()[0], y=df.keys()[1::],  colormap='jet', legend=False)
        ymin,ymax = ax.get_ylim()
        skill = 0
        for tag, start, end in  tag_stime:
            if int(tag) != 0:
                if int(tag) < 0:
                    print ('Ignore the anomaly tag < 0')
                    continue
                elif int(tag) == 9:
                    print ('Ignore the tag == 9')
                    continue
                else:
                    c = 'green'
                ax.axvline(start, c=c)
                skill += 1
                ax.text(start, ymax-0.05*(ymax-ymin), 'skill %s'% skill, color=c, rotation=-90)
            else:
                print ('Ignore the useless tag=0')
                continue
                ax.axvline(start, c = 'gray')
        ax.set_ylabel("Scaled Value")
    axarr[nfiles-1].set_xlabel("Time(secs)")
    output_dir = os.path.join(dir_of_this_script, 'plots')        
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    f.savefig(os.path.join(output_dir, "multi_whole_success_exps.png"),format='png', dpi=300)
    plt.show()
        
