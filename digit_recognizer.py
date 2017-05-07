"""
This project does the digit_recoginzer from the Kaggle project
@author: Bala Bathula, bgsquare@gmail.com
Each digit is represented as the pixel of size 28 x 28 =784 pixel image

"""
import pandas as pd
import os

# Load the dataset

def get_data(train=True,base_dir='data'):
    """
    This function gets the train/test data. By default the train=True. If train=False, then the test data would be returned
    output is a dataframe (train or test)
    """
    
    if train==True:
        return pd.read_csv(os.path.join(base_dir,'train.csv'))
        #return pd.read_csv(os.path.join(base_dir,'ex3_data.csv')) # Use this for Coursera excercise
    else:
        return pd.read_csv(os.path.join(base_dir,'test.csv'))
    
    

if __name__=="__main__":
    print("Starting the Digit recoginzer Kaggle compititions.....")
    