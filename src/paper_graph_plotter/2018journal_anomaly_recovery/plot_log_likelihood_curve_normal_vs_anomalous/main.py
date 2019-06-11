from sklearn.externals import joblib
from birl_hmm.hmm_training import hmm_util
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import ipdb
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

def rgba_to_rgb_using_white_bg(rgb_array, alpha):
    ret = [i*alpha+(1-alpha) for i in rgb_array]+[1]
    return ret


if __name__ == '__main__':
    model = joblib.load("models/skill 5/introspection_model") 
    hmm_model = model['hmm_model']

    list_of_succ_mat = []
    for path in glob.glob("successful_skills/skill 5/*/*.csv"):
        with open(path, 'r') as f:
            df = pd.read_csv(f)
            list_of_succ_mat.append(df.values[:, 1:])

    list_of_unsucc_mat = []
    for path in glob.glob("unsuccessful_skills/skill 5/*/*.csv"):
        with open(path, 'r') as f:
            df = pd.read_csv(f)
            list_of_unsucc_mat.append(df.values[:, 1:])

    list_of_succ_log_curves = [hmm_util.fast_log_curve_calculation(i, hmm_model) for i in list_of_succ_mat]
    list_of_unsucc_log_curves = [hmm_util.fast_log_curve_calculation(i, hmm_model) for i in list_of_unsucc_mat]



    list_of_processed_succ_curve = []
    for ys in list_of_succ_log_curves:
        list_of_processed_succ_curve.append(np.interp(np.linspace(0, len(ys), 100), range(len(ys)), ys)/(len(ys)/100.0))


    succ_mean = np.mean(list_of_processed_succ_curve, axis=0)
    succ_std = np.std(list_of_processed_succ_curve, axis=0)
    succ_upper_bound = succ_mean+2*succ_std
    succ_lower_bound = succ_mean-2*succ_std
    

    list_of_processed_unsucc_curve = []
    for ys in list_of_unsucc_log_curves:
        list_of_processed_unsucc_curve.append(np.interp(np.linspace(0, len(ys), 100), range(len(ys)), ys)/(len(ys)/100.0))


    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.fill_between(range(len(succ_upper_bound)), succ_lower_bound, succ_upper_bound, color=rgba_to_rgb_using_white_bg((0,0,1), 0.25))

    for ys in list_of_processed_succ_curve:
        ax.plot(ys, color=rgba_to_rgb_using_white_bg((0,0,1), 0.7))

    for ys in list_of_processed_unsucc_curve:
        ax.plot(ys, color='red')

    ax.plot(succ_mean, color='black', linewidth=3)
    

    ax.set_ylim(bottom=-100, top=4500)
    ax.set_xlim(-1, 99)

    blue_patch = mpatches.Patch(color=rgba_to_rgb_using_white_bg((0,0,1), 0.25), label='Normal trials: mean+-2*std')
    black_line = mlines.Line2D([], [], color='black', linewidth=4, label="Normal trials: mean")
    succ_line = mlines.Line2D([], [], color=rgba_to_rgb_using_white_bg((0,0,1), 0.7), label="Normal trials")
    red_line = mlines.Line2D([], [], color='red', label="Anomalous trials")
    ax.legend(handles=[succ_line, red_line, black_line, blue_patch])
    
    ax.set_ylabel("Log-likelihood")
    ax.set_xlabel("Time(s)")
    ax.set_title("Log-likelihood Curves: Normal VS Anomalous Trials")

    fig.savefig('log-likelihood curves.eps', format='eps', dpi=300)
    fig.savefig('log-likelihood curves.png', format='png', dpi=300)    
    plt.show()
