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

def select_tag_number_occu(data):
    output=[]
    for i in data:
        if not i in output:
            output.append(i)
    return output


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
        all_time = np.array(df.iloc[:, 0])

        tag_ranges = pickle.load(open(os.path.join(os.path.dirname(csv), 'tag_ranges.pkl'), 'rb'))
        tag_stime = []
        tag_set = []
        for tag, (st, et) in tag_ranges:
            tag_stime.append((tag, st.to_sec()-s_t, et.to_sec()-s_t))
            if int(tag) > 0 and int(tag) < 1000:
                tag_set.append(int(tag))
        tag_set_select = select_tag_number_occu(tag_set)

        #     if tag >=0:
        #         if int(tag) > 0 and int(tag) < 1000:
        #             tag_set.append(int(tag))
        #     else:
        #         tag_set=[]
        #         break
        # tag_set_select = select_tag_number_occu(tag_set)
        
        if len(tag_set)==6 and len(tag_set_select)==6:
            for tag, start, end in tag_stime:
                if int(tag) != 0:
                    if int(tag) <= 0 or int(tag) >= 1000:
                        print ('Ignore the anomaly tag <= 0 or tag >= 1000')
                        continue
                # if int(tag) < 0:
                #     print ('Ignore the anomaly tag < 0 or tag >= 1000')
                #     continue
                    else:
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
                        values = tag_df[select_list].values
                        if len(values) !=0:
                            values[:,23] = preprocessing.minmax_scale(values[:,23],feature_range=(0,1))
                            values[:,24] = preprocessing.minmax_scale(values[:,24],feature_range=(0,1))

                            # for ii,value in enumerate(values):
                            #     values_count[ii] = np.append(values)

                            values_tag = np.insert(values,0,values=tag, axis=1)
                            values_tags.extend(values_tag)

            values_tags_trial_num = np.insert(values_tags,0,values=i, axis=1)
            values_tags_trial_nums.append(values_tags_trial_num)

        else:
            print ('It is not a whole frequency completed experiments which without any recoveryskill ')

    values_tags_trial_nums_clean=values_tags_trial_nums
    np.save("data/whole_complete_no_recovery.npy", values_tags_trial_nums_clean)

    print(np.shape(values_tags_trial_nums_clean))
