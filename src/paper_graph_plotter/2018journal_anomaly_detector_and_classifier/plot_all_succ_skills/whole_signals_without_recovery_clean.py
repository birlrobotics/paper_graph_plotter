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
    data_path = "/home/birl-spai-ubuntu14/baxter_ws/src/SPAI/smach_based_introspection_framework/jim_folder_have_tag_range/anomaly_detection_feature_selection_folder/No.0 filtering scheme/whole_experiment/"
    experience_tuple = []
    experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
    sensor_info=[]
    tag_info=[]
    sensor_info_et=[]
    tag_info_et=[]
    values_tags_trial_nums=[]


    for i, csv in enumerate(glob.glob(os.path.join(data_path, "*", "*csv"))):
        values_tags=[]

        df = pd.read_csv(csv)
        s_t = df.iloc[0, 0]
        df.iloc[:, 0] = df.iloc[:, 0]-s_t

        tag_ranges = pickle.load(open(os.path.join(os.path.dirname(csv), 'tag_ranges.pkl'), 'rb'))
        tag_stime = []
        for tag, (st, et) in tag_ranges:
            tag_stime.append((tag, st.to_sec()-s_t, et.to_sec()-s_t))

        for tag, start, end in tag_stime:
            if int(tag) != 0:
                if int(tag) <= 0 or int(tag) >= 1000:
                    print ('Ignore the anomaly tag <= 0 or tag >= 1000')
                    continue
                # if int(tag) < 0:
                #     print ('Ignore the anomaly tag < 0 or tag >= 1000')
                #     continue
                else:
                    # tag_df: extract the data within the time range (start, end)
                    tag_df = df.loc[(df['Unnamed: 0'] >= start) & (df['Unnamed: 0'] <= end)]
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
                    # extract the specific data from the tag_df
                    values = tag_df[select_list].values
                    # sensor_values = df[select_list].values
                    # if len(values) !=0:
                    #     state = values[0]
                    #     next_state = values[-1]
                    # else:
                    #     continue
                 
                    # action = tag
                    # if tag == 8:
                    #     reward = 100
                    #     done = True
                    # else:
                    #     reward = -1
                    #     done = False
                    # e = experience(state, action, reward, next_state, done)
                    # experience_tuple.append(e)
                    # sensor_info.extend(sensor_values)
                    sensor_info.extend(values)
                    for ii in range(len(values)):
                        tag_info.extend([tag])
                    
                    values_tag = np.insert(values,0,values=tag, axis=1)
                    values_tags.extend(values_tag)
        values_tags_trial_num = np.insert(values_tags,0,values=i, axis=1)
        values_tags_trial_nums.extend(values_tags_trial_num)
        # if i > 100:
        #     break
    values_tags_trial_nums_clean = []
    values_tags_trial_nums_thinnner=[]
        # sensor_info_et.append(next_state)
        # tag_info_et.extend([action])
    for i, data in enumerate(values_tags_trial_nums):
        if data[21] > 0:
            data[21] = 1
        if data[22] > 0:
            data[22] = 1
        values_tags_trial_nums_clean.append(data)

        # if i % 3 ==1:
        #     values_tags_trial_nums_thinnner.append(data)


                        
        # if i > 300: break
    # np.save("experience_tuple_no_recovery_skill_pos.npy", experience_tuple)
    # np.save("experience_tuple_no_recovery_skill_7dim_500.npy", experience_tuple)
    # np.save("experience_tuple_no_recovery_skill_7dim_50000.npy", experience_tuple)
    # np.save("sensor_info_no_recovery_skill_pos.npy", sensor_info)
    # np.save("tag_info_no_recovery_skill_pos.npy", tag_info)
    # np.save("sensor_info_no_recovery_skill_pos_20dim_recovery.npy", sensor_info)
    # np.save("tag_info_no_recovery_skill_pos_20dim_recovery.npy", tag_info)
    # np.save("tag_info_no_recovery_skill_pos_21dim_recovery.npy", tag_info)
    np.save("tag_info_no_recovery_skills_27dim_recovery_values_tags_trial_nums_clean.npy", values_tags_trial_nums_clean)
    # np.save("tag_info_no_recovery_skills_27dim_recovery_values_tags_trial_nums_thinner.npy", values_tags_trial_nums_thinnner)


    print(np.shape(sensor_info))
    print(np.shape(tag_info))
    print(np.shape(values_tags_trial_nums))
    print(values_tags_trial_nums[0])
    print(values_tags_trial_nums[7000])

    print(np.shape(values_tags_trial_nums_clean))
    # print(np.shape(values_tags_trial_nums_thinnner))

    # np.save("sensor_info_no_recovery_skill_pos_et.npy", sensor_info_et)
    # np.save("tag_info_no_recovery_skill_pos_et.npy", tag_info_et)
        # f = open('experience_tuple_no_recovery_skill_pos.txt','w')
        # f.write(str(experience_tuple))
        # f.close()