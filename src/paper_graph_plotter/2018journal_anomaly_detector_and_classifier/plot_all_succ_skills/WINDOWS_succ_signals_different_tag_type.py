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
from collections import namedtuple
import ipdb
coloredlogs.install()

if __name__ == '__main__':
    logger = logging.getLogger()

    dir_of_this_script = os.path.dirname(os.path.realpath(__file__))
    
    a = 0
    for skill in (3,4,5,7,8,9,1000,1001,1002,1003):
        data_path = '/home/birl-spai-ubuntu14/baxter_ws/src/SPAI/smach_based_introspection_framework/jim_folder_have_tag_range/anomaly_detection_feature_selection_folder/No.0 filtering scheme/successful_skills/skill ' + str(skill)

        signals=[]
        label_occu=[]
        for i, csv in enumerate(glob.glob(os.path.join(data_path, "*", "*csv"))):
            print(i)
            print(csv)
    #        ax.set_title("Multimodal signals of norminal execution #%s" %(i+1))
    #       
            df = pd.read_csv(csv)
            s_t = df.iloc[0, 0]
            e_t = df.iloc[-1, 0]
            
            df.iloc[:, 0] = df.iloc[:, 0]-s_t
            all_time = np.array(df.iloc[:, 0])

            anomaly_type='normal'
            experiment=i
            label_occu.append([experiment, skill, anomaly_type])

            select_list = ['baxter_enpoint_pose.pose.position.x',# position
                            'baxter_enpoint_pose.pose.position.y',
                            'baxter_enpoint_pose.pose.position.z',

                            'baxter_enpoint_pose.pose.orientation.x', # orientation
                            'baxter_enpoint_pose.pose.orientation.y',
                            'baxter_enpoint_pose.pose.orientation.z',
                            'baxter_enpoint_pose.pose.orientation.w',

                            'baxter_enpoint_pose.twist.angular.norm', # angular
                            'baxter_enpoint_pose.twist.angular.x',
                            'baxter_enpoint_pose.twist.angular.y',
                            'baxter_enpoint_pose.twist.angular.z',

                            'baxter_enpoint_pose.twist.linear.norm', # linear
                            'baxter_enpoint_pose.twist.linear.x',
                            'baxter_enpoint_pose.twist.linear.y',
                            'baxter_enpoint_pose.twist.linear.z',
                            
                            'robotiq_force_sensor.wrench.force.norm', # force
                            'robotiq_force_sensor.wrench.torque.norm', # force
                            'wrench.force.x',
                            'wrench.force.y',
                            'wrench.force.z',
                            'wrench.torque.x',
                            'wrench.torque.y',
                            'wrench.torque.z',

                            'tactile_static_data.left.std', #tactile
                            'tactile_static_data.right.std'
                            ]
            values = df[select_list].values
            values_clean=[]
            for data in values:
                if data[23] > 0:
                    data[23] = 1
                if data[24] > 0:
                    data[24] = 1
                values_clean.append(data)

            values_count = []
            for count, value in enumerate(values_clean):
                value_c = np.append(value,all_time[count])
                values_count.append(value_c)
            # values_count = np.append(values, all_time, axis=1)
            values_trials = np.insert(values_count,0,values=i, axis=1)


            signals.append(values_trials)
        values_tags_trial_nums_clean = np.array(signals)
        label_occu = np.array(label_occu)


            # if i > 300: break
        np.save("success_unsuccess/windows/success_skill_"+ str(skill)+"_27dim_windows.npy", values_tags_trial_nums_clean)
        np.save("success_unsuccess/windows/success_skill_"+ str(skill)+"_27dim_windows_label_occu.npy", label_occu)

        print(np.shape(values_tags_trial_nums_clean))
        print(np.shape(label_occu))
        a+=len(values_tags_trial_nums_clean)
    print(a)