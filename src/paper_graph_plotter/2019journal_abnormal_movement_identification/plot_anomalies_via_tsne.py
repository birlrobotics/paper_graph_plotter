from sklearn.manifold import TSNE
import glob
import os,ipdb
import pandas as pd
import pprint
import coloredlogs, logging
import sys, traceback
from sklearn.externals import joblib
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import OrderedDict

# plt.rcParams.update({'font.size': 12})
# plt.rcParams["font.family"] = "Time New Roman"


coloredlogs.install()
pp = pprint.PrettyPrinter(indent=4)

def run():
    start = time.time()
    logger = logging.getLogger('GenClassificationModels')

    succ_folder = "/home/birl_wu/baxter_ws/src/SPAI/smach_based_introspection_framework/introspection_data_folder.AC_offline_test/anomaly_detection_feature_selection_folder/No.0 filtering scheme/successful_skills" 
    succ_skills_csvs = glob.glob(os.path.join(
        succ_folder,
        '*',
        '*',
        '*.csv',
    ))
    
    anomaly_classification_feature_selection_folder = "/home/birl_wu/baxter_ws/src/SPAI/smach_based_introspection_framework/introspection_data_folder.AC_offline_test/anomaly_classification_feature_selection_folder/" 
    folders = glob.glob(os.path.join(
        anomaly_classification_feature_selection_folder,
        'No.* filtering scheme',
        'anomalies_grouped_by_type',
        'anomaly_type_(*)',
    ))
    
    list_of_labels = []
    list_of_type = []
    atype = 1000 # nominal 

    datapoints = None

    '''
    # load nominal data
    for csv in succ_skills_csvs:
        logger.info(csv)
        df = pd.read_csv(csv, sep=',')
        # Exclude 1st column which is time index
        mat = df.values[:,1:]
        datapoints = mat if datapoints is None else np.vstack((datapoints, mat))
        list_of_labels += ['nominal']*mat.shape[0]
        list_of_type += [atype]*mat.shape[0]
        
    '''
    logger.warning("finished loading the nominal data, Ganna to load the abnormal data")
    
    # load abnormal data        
    atype = 0
    for folder in folders:
        logger.info(folder)
        path_postfix = os.path.relpath(folder, anomaly_classification_feature_selection_folder).replace("anomalies_grouped_by_type"+os.sep, "")
        anomaly_type = re.compile(r"\((.+)\)").search(path_postfix).group(1)
        csvs = glob.glob(os.path.join(
            folder,
            '*', '*.csv',
        ))
        
        for j in csvs:
            df = pd.read_csv(j, sep=',')
            # Exclude 1st column which is time index
            mat = df.values[:,1:]
            datapoints = mat if datapoints is None else np.vstack((datapoints, mat))
            list_of_labels += [anomaly_type]*mat.shape[0]
            list_of_type += [atype]*mat.shape[0]
        atype += 1
    
    logger.warning("data loaded, Ganna to tsne")
    
    colors = {1000: "black",
              0:'r',
              1:'g',
              2:'b',
              3:'k',
              4:'c',
              5:'m',
              6:'gold',
              7:'gray',}
        
    for i in range(20):    
        tsne = TSNE(n_components = 2, learning_rate = 100, verbose = 2).fit_transform(datapoints)
        logger.warning("Transformed, Ganna to plot")
        plt.figure()
        for atype, label, point in zip(list_of_type, list_of_labels, tsne):
            plt.scatter(point[0], point[1], c = colors[atype], label=label)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc=1)
        plt.savefig('tsne#%s.png'%i, format="png", dpi=300)
        plt.savefig('tsne#%s.eps'%i, format="eps", dpi=300)    
        end = time.time()
        logger.warning(end - start)
    
    plt.show()
    
if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    logger.addHandler(consoleHandler)
    run()
    

