import os, sys
import glob
import itertools
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import bnpy
import ipdb
def run():
    folders = glob.glob(os.path.join(sys.path[0], 'latest','*','*_model'))
    for folder in itertools.chain(folders):
        model_type = os.path.basename(os.path.dirname(folder))
        model = joblib.load(folder)
        print folder
        print model.keys()
        try:
            hmm_model = model['hmm_model']
        except:
            print ('Cannot load the specific hmm_model:%s'%model_type)
        ipdb.set_trace()

if __name__=='__main__':
    run()
