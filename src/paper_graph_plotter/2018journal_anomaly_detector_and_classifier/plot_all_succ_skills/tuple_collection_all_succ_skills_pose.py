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
    data_path = "/home/birl-spai-ubuntu14/baxter_ws/src/SPAI/smach_based_introspection_framework/birl_anomaly_dataset/anomaly_detection_feature_selection_folder/No.0 filtering scheme/whole_experiment"
    experience_tuple = []
    experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])

    for i, csv in enumerate(glob.glob(os.path.join(data_path, "*", "*csv"))):
        df = pd.read_csv(csv)
        s_t = df.iloc[0, 0]
        df.iloc[:, 0] = df.iloc[:, 0]-s_t

        tag_ranges = pickle.load(open(os.path.join(os.path.dirname(csv), 'tag_ranges.pkl'), 'rb'))
        tag_stime = []
        for tag, (st, et) in tag_ranges:
            tag_stime.append((tag, st.to_sec()-s_t, et.to_sec()-s_t))
        

        for tag, start, end in tag_stime:
            if int(tag) != 0:
                if int(tag) < 0 or int(tag) >= 1000:
                    print ('Ignore the anomaly tag < 0 or tag >= 1000')
                    continue
                # if int(tag) < 0:
                #     print ('Ignore the anomaly tag < 0 or tag >= 1000')
                #     continue
                else:
                    tag_df = df.loc[(df['Unnamed: 0'] >= start) & (df['Unnamed: 0'] <= end)]
                    select_list = ['baxter_enpoint_pose.pose.position.x',# position
                                    'baxter_enpoint_pose.pose.position.y',
                                    'baxter_enpoint_pose.pose.position.z',

                                #    'baxter_enpoint_pose.pose.orientation.x', # orientation
                                #    'baxter_enpoint_pose.pose.orientation.y',
                                #    'baxter_enpoint_pose.pose.orientation.z',
                                #    'baxter_enpoint_pose.pose.orientation.w',

                                #    'baxter_enpoint_pose.twist.angular.norm', # angular
                                #    'baxter_enpoint_pose.twist.angular.x',
                                #    'baxter_enpoint_pose.twist.angular.y',
                                #    'baxter_enpoint_pose.twist.angular.z',

                                #    'baxter_enpoint_pose.twist.linear.norm', # linear
                                #    'baxter_enpoint_pose.twist.linear.x',
                                #    'baxter_enpoint_pose.twist.linear.y',
                                #    'baxter_enpoint_pose.twist.linear.z',
                                
                                #    'robotiq_force_sensor.wrench.force.norm', # force
                                #    'robotiq_force_sensor.wrench.torque.norm', # force

                                #    'wrench.force.x',
                                #    'wrench.force.y',
                                #    'wrench.force.z',
                                #    'wrench.torque.x',
                                #    'wrench.torque.y',
                                #    'wrench.torque.z'
                                ]

                    values = tag_df[select_list].values
                    
                    position_list = ['baxter_enpoint_pose.pose.position.x',# position
                                'baxter_enpoint_pose.pose.position.y',
                                'baxter_enpoint_pose.pose.position.z',
                                ]
                    # state = values[0]
                    # next_state = values[-1]
                    if len(values) !=0:
                        state = values[0]
                        next_state = values[-1]
                    else:
                        continue
                 
                    action = tag
                    if tag == 8:
                        reward = 100
                        done = True
                    else:
                        reward = -1
                        done = False
                    e =  experience(state, action, reward, next_state, done)
        experience_tuple.append(e)
        

                        
        # if i > 300: break
    np.save("experience_tuple_no_recovery_skill_positions.npy", experience_tuple)
