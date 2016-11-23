'''
Visualizations modules for the digit recogizer
@author: Bala Bathula
'''
import pandas
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from digit_recognizer import get_data
import math
import os

def display_sample_data(df,n_samples=100,random_points=True,outdir='plots'):
# def display_sample_data(df,**kwargs):
    """
    Display a sample of data in the training data set
    n_samples =Number of samples to display (this will be usually in n*n form)
    Most of the logic is borrowed from Coursera ML Octave code
    """
    try:
        os.makedirs(outdir);
    except:
        pass
    
    m_length=df.shape[0]; # Number of rows in the data
    pixel_length=df.shape[1]; # Number of pixels in the data; this need not be in n*n form...we could have 10 x 15 pixel image as well
    
    sample_height=int(round(math.sqrt(pixel_length)))
    sample_width=int(pixel_length/sample_height)
    
    
    plt_rows=int(math.floor(math.sqrt(n_samples)))
    plt_cols=int(math.ceil(n_samples/plt_rows))
    
    
    
    # Padding between the images
    pad=1
    
    # Setup a blank display
    display_array=-np.ones((pad+plt_rows*(sample_height+pad),pad+plt_rows*(sample_width+pad)));
    
    if random_points==True:
        randindices=np.random.randint(m_length,size=n_samples)# Select some n random samples from m data points (n<=m)
    else: # Select only first n points of the data
        randindices=np.arange(0,n_samples);
         
    sample_df=df.ix[randindices] # .ix command takes indices as the argument and the dataframe is created
    # Drop the labels from the sample_df (these are digit numbers)
    sample_df=sample_df.drop('label',axis=1)
    
    # Convert the sample dataframe to an array of image patches
    # i.e., copy each example into a patch on the display array
    
    curr_sample=0;
    for i in range(plt_rows):
        for j in range(plt_cols):
            if curr_sample > n_samples-1:
                break
            # Copy the patch of the pixel data
            sample_df_index=randindices[curr_sample]; # This is the index coming from 
            # max_value=max(abs(sample_df.ix[sample_df_index,:])) # Take the maximum value absolute value across all the columns
            # Update the display array with the reshaped matrix
            tmp_array=np.array(sample_df.ix[sample_df_index,:])
            
            
            # print tmp_array
            row_ll=pad+i*(sample_height+pad)+0
            row_ul=pad+i*(sample_height+pad)+sample_height
            col_ll=pad+j*(sample_width+pad)+0
            col_ul=pad+j*(sample_width+pad)+sample_width
            # display_array[row_ll:row_ul,col_ll:col_ul]=np.reshape(tmp_array, (sample_height,sample_width))/max_value
            display_array[row_ll:row_ul,col_ll:col_ul]=np.reshape(tmp_array, (sample_height,sample_width))
            curr_sample+=1; # Increment
        # Another break statement outside for loop    
        if curr_sample > n_samples-1:
                break
    plt.style.use('ggplot');
    fig=plt.figure() # Create a fig handler
    ax=fig.add_subplot(111)
    ax.imshow(display_array,cmap=cm.binary)
    plt.axis('off')
    fig.savefig(os.path.join(outdir,'grid_digits.eps'))
    plt.show()
    
    return


if __name__=='__main__':
    print('Creates visualizations for the digit recognizer...')
    display_sample_data(get_data(),random_points=False,n_samples=100)
    