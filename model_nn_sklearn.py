"""
This one uses the sklearn library for neural network
@author: Bala Bathula
Multi-layer perceptron classifier
"""
print(__doc__)


from sklearn.neural_network  import MLPClassifier
from digit_recognizer import get_data
import matplotlib.pyplot as plt
import numpy as np



def test():
    """
    Use this for testing the Multi-layer perceptron classifier
    """
    X=[[0,0],[1,1]]
    y=[[0,1],[1,1]]
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,), random_state=1); # (5,) represents 5 units in each hidden layer
    # Number of hidden layers = len((5,2)) say. Which means  there are two hidden layers and activation units are 5 and 2 respectively
    # If nothing is specified in the second-tuple then the number of hidden layers is just 1
    
    clf.fit(X,y)
     
    
    
    return


if __name__=='__main__':
    
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import os
    
    print ('Starting the neural network model for Digit recognizer using sklearn ....')
    
    full_data_df=get_data(); # The data-structure here is pandas dataframe
    test_df=get_data(train=False)
    
    full_data_matrix=full_data_df.as_matrix()
    test_matrix=test_df.as_matrix()
    
    y=full_data_matrix[:,0]
    
    X=full_data_matrix[:,1:]
    # Normalize the full data
    X=X*1.0/np.max(X);
    
    # Normalize the test data
    X_test=test_matrix*1.0/np.max(test_matrix)
    
    
    # Split the data into train and cross-validation set
    X_train, X_cv, y_train, y_cv = train_test_split(X, y,train_size=0.7,random_state=13)
    
    solver='sgd'
    activation='tanh' # Default is 'relu'
    hidden_layer_size=(50,); # Mean one hidden layer with 50 activation units
    nbr_hidden_layers=len(hidden_layer_size);
    
    mlp = MLPClassifier(activation=activation,hidden_layer_sizes=hidden_layer_size, max_iter=10, alpha=1e-4,
                    solver=solver, verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
    
    mlp.fit(X_train, y_train)
    print("Training set score: %f" % mlp.score(X_train, y_train))
    print("Cross-validation set score: %f" % mlp.score(X_cv, y_cv))
    print(mlp.coefs_)
    weights_shape=[i.shape for i in mlp.coefs_]
    print (weights_shape)
    fig, axes = plt.subplots(4, 4)
    # use global min / max to ensure all weights are shown on the same scale
    vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
    for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
        ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
                   vmax=.5 * vmax)
        ax.set_xticks(())
    
        ax.set_yticks(())

    plt.show()
    fig.savefig(os.path.join('plots','nn_weights_%s_%s.eps'%(solver,activation)))
    outdir='results'
    try:
        os.makedirs(outdir)
    except:
        pass
    
    test_pred_digits=mlp.predict(X_test)
    df_test_pred=pd.DataFrame(test_pred_digits,index=range(1,test_pred_digits.shape[0]+1),columns=['Label'])
    df_test_pred.index.names=['ImageId']
    
    df_test_pred.to_csv('%s/nn_kaggle_test_results_sklearn_%s_%s_%d.csv'%(outdir,solver,activation,nbr_hidden_layers))