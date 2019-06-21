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


# SKILL3={'human_collision'}
# SKILL4={'human_collision','tool_collision'}
# SKILL5={'human_collision'}
# SKILL7={'human_collision'}
# SKILL8={'human_collision'}
# SKILL9={'human_collision'}

SKILL={
    # 3:{'human_collision','no_object'},
    3:{'human_collision'},
    4:{'human_collision','tool_collision','no_object'},
    5:{'object_slip','no_object','human_collision'},
    7:{'object_slip','wall_collision','human_collision'},
    8:{'human_collision','tool_collision','no_object'},
    # 9:{'human_collision'},
    1000:{'tool_collision'},
    1001:{'human_collision','tool_collision','object_slip'},
    # 1002:{'Unlabeled'},
    1003:{}
}



if __name__ == '__main__':
    logger = logging.getLogger()

    dir_of_this_script = os.path.dirname(os.path.realpath(__file__))
    
    a = 0
    for skill in (3,4,5,7,8,1000,1001):
        data_path = '/home/birl-spai-ubuntu14/baxter_ws/src/SPAI/smach_based_introspection_framework/jim_folder_have_tag_range_100hz/anomaly_detection_feature_selection_folder/No.0 filtering scheme/unsuccessful_skills/skill ' + str(skill)
        anomaly_names = SKILL[skill]

        signals=[]
        label_occu=[]
        for i, csv in enumerate(glob.glob(os.path.join(data_path, "*", "*csv"))):
            print(i)
            print(csv)
    #        ax.set_title("Multimodal signals of norminal execution #%s" %(i+1))
    #       
            anomaly_type, anomaly_gentime = pickle.load(open(os.path.join(os.path.dirname(csv), 'anomaly_label_and_signal_time.pkl'), 'rb'))
            df = pd.read_csv(csv)
            s_t = df.iloc[0, 0]
            e_t = df.iloc[-1, 0]
            
            df.iloc[:, 0] = df.iloc[:, 0]-s_t
            all_time = np.array(df.iloc[:, 0])

            anomaly_time = anomaly_gentime.to_sec()-s_t
            if anomaly_type in anomaly_names and anomaly_time > 0:

                experiment=i

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
                if len(values) > 10:
                    values[:,23] = preprocessing.minmax_scale(values[:,23],feature_range=(0,1))
                    values[:,24] = preprocessing.minmax_scale(values[:,24],feature_range=(0,1))
                    values_clean = values
                    values_count = []
                    for count, value in enumerate(values_clean):
                        value_c = np.append(value,all_time[count])
                        values_count.append(value_c)
                    # values_count = np.append(values, all_time, axis=1)
                    values_trials = np.insert(values_count,0,values=experiment, axis=1)

                    label_occu.append([experiment, skill,anomaly_type, anomaly_time])

            signals.append(values_trials)
        values_tags_trial_nums_clean = np.array(signals)
        label_occu = np.array(label_occu)

        np.save("data/windows/100hz_unsuccess_skill_"+ str(skill)+"_27dim_windows.npy", values_tags_trial_nums_clean)
        np.save("data/windows/100hz_unsuccess_skill_"+ str(skill)+"_27dim_windows_label_occu.npy", label_occu)

        print(np.shape(values_tags_trial_nums_clean))
        print(np.shape(label_occu))
        a+=len(values_tags_trial_nums_clean)
    print(a)