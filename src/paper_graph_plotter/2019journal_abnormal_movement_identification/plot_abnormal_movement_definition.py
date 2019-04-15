import os, ipdb
import glob               
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame

if __name__=="__main__":    
    data_path = os.path.dirname(os.path.realpath(__file__))
    skill = 3

    fig, axarr = plt.subplots(nrows=1, ncols=3,  figsize=(18,3))
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')    

    image = plt.imread('lift_up_movement.png')
    im = axarr[0].imshow(image)

    # load successful dataset for training and validating
    succ_csv = glob.glob(os.path.join(
        data_path,
        'successful_skills',
        'skill %s'%skill,
        '*',
        '*.csv'
    ))[1]
      
    succ_df = read_csv(succ_csv, header=0, index_col=0)
    succ_df.plot(ax = axarr[1], legend= False)

    unsucc_csv = glob.glob(os.path.join(
        data_path,
        'unsuccessful_skills',
        'skill %s'%skill,
        '*',
        '*.csv'
    ))[0]

    unsucc_df = read_csv(unsucc_csv, header=0, index_col=0)    
    unsucc_df.plot(ax = axarr[2], legend=False)
    fig.savefig('lift_up.eps', format = 'eps', dpi=300)
    fig.savefig('lift_up.png', format = 'png', dpi=300)    
    plt.show()
