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
from sklearn import preprocessing

coloredlogs.install()



if __name__ == '__main__':
    logger = logging.getLogger()
    dir_of_this_script = os.path.dirname(os.path.realpath(__file__))
    data_path = "/home/birl-spai-ubuntu14/baxter_ws/src/SPAI/smach_based_introspection_framework/jim_folder_have_tag_range/anomaly_detection_feature_selection_folder/No.0 filtering scheme/whole_experiment/"
    experience_tuple = []
    experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
    sensor_info=[]
    tag_info=[]
    sensor_info_et=[]
    tag_info_et=[]
    # values_tags=[]
    values_tags_trial_nums=[]


    for i, csv in enumerate(glob.glob(os.path.join(data_path, "*", "*csv"))):
        values_tags_nums=[]
        df = pd.read_csv(csv)
        s_t = df.iloc[0, 0]
        df.iloc[:, 0] = df.iloc[:, 0]-s_t

        tag_ranges = pickle.load(open(os.path.join(os.path.dirname(csv), 'tag_ranges.pkl'), 'rb'))

        tag_stime = []
        for tag, (st, et) in tag_ranges:
            tag_stime.append((tag, st.to_sec()-s_t, et.to_sec()-s_t))

        last_end = 0
        for tag, start, end in tag_stime:
            Times=[]
            if int(tag) != 0:
                # if int(tag) < 0 or int(tag) >= 1000:
                #     print ('Ignore the anomaly tag < 0 or tag >= 1000')
                #     continue
                if int(tag) <= 0:
                    print ('Ignore the anomaly tag <= 0')
                    continue
                else:
                    # tag_df: extract the data within the time range (start, end)
                    tag_df = df.loc[(df['Unnamed: 0'] >= start) & (df['Unnamed: 0'] <= end)]
                    tag_start = 0
                    tag_end = end - start

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
                    # extract the specific data from the tag_df
                
                    last_start =start
                    Times = np.append(Times,last_end) ### ground times to sequens which have usdful skill
                    for T in range(len(tag_df)):
                        ground_time = tag_df.iloc[T,0] - last_start + last_end
                        Times = np.append(Times,ground_time)
                    last_end=[Times[-1]]

                    values = tag_df[select_list].values
                    if len(values) >= 10:
                        values[:,23] = preprocessing.minmax_scale(values[:,23],feature_range=(0,1))
                        values[:,24] = preprocessing.minmax_scale(values[:,24],feature_range=(0,1))
                        
                        values_count=[]
                        for ii, value in enumerate(values):
                            value = np.append(value,Times[ii])
                            value = np.append(value,ii)
                            values_count.append(value)
                        
                        values_tags = np.insert(values_count,0,values=tag, axis=1)
                        values_tags_num = np.insert(values_tags,0,values=i, axis=1)
                        values_tags_nums.append(values_tags_num)
                        Times=[]

        values_tags_trial_nums.extend(values_tags_nums)

    values_tags_trial_nums_clean = values_tags_trial_nums
    np.save("data/windows_have_recovery_skills_29dim_recovery_values_tags_trial_nums_clean.npy", values_tags_trial_nums_clean)

    print(np.shape(values_tags_trial_nums_clean))
