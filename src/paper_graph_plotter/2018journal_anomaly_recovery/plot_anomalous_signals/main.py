import glob
import os
import coloredlogs, logging
import pickle
import matplotlib.pyplot as plt
import pandas as pd

coloredlogs.install()

if __name__ == '__main__':
    logger = logging.getLogger("plot_anomalous_signals.main")
    dir_of_this_script = os.path.dirname(os.path.realpath(__file__))
    for csv in glob.glob(os.path.join(dir_of_this_script, "data", "*", "*csv")):
        relpath = os.path.relpath(csv, os.path.join(dir_of_this_script))
        logger.info(relpath) 


        anomaly_type, anomaly_gentime = pickle.load(open(os.path.join(os.path.dirname(csv), "anomaly_label_and_signal_time.pkl"), 'rb')) 

        df = pd.read_csv(csv)
        s_t = df.iloc[0, 0]
        df.iloc[:, 0] = df.iloc[:, 0]-s_t
        anomaly_time = anomaly_gentime.to_sec()-s_t

        for dim in df.columns[1:]:
            output_dir = os.path.join(dir_of_this_script, "plots", "anomaly type: %s"%anomaly_type, "dimension: %s"%dim)
            logger.info(dim)
            fig, ax = plt.subplots(nrows=1, ncols=1)


            ax.plot(df.iloc[:, 0], df[dim])
            ax.axvline(anomaly_time, c='red')

            xmin,xmax = ax.get_xlim()
            ymin,ymax = ax.get_ylim()
            ax.text(anomaly_time+0.01*(xmax-xmin), ymax-0.05*(ymax-ymin), 'anomaly signal', color='red', rotation=-90)

            ax.set_xlabel("seconds")
            if 'twist.angular' in dim:
                ax.set_ylabel("radians per second")
                data_type = "norm of angular velocity" 
            elif 'twist.linear' in dim:
                ax.set_ylabel("meters per second")
                data_type = "norm of linear velocity" 
            elif 'wrench.force' in dim:
                ax.set_ylabel("N")
                data_type = "norm of force" 
            elif 'wrench.torque' in dim:
                ax.set_ylabel("N*m")
                data_type = "norm of torque" 
            elif 'tactile_dynamic_data' in dim:
                data_type = "tactile sensor dynamic daya" 
            elif 'tactile_static_data' in dim:
                data_type = "tactile sensor static data"

            ax.set_title("anomaly type: %s, data type: %s"%(anomaly_type, data_type))

            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            fig.savefig(os.path.join(output_dir, "%s.eps"%os.path.basename(csv)[:-4]))
            plt.close(fig)
