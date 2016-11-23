'''
This is the ML model file for logistic regression method
@author Bala Bathula
'''

import numpy as np
import scipy.optimize as spo
import os
import matplotlib.pyplot as plt

J_cost_all=[]; # This will be used for plotting the cost function for very label, for every iteration
global label_index; # This is necessary, because we increment the variable in the function. If it is not declared, python thinks it is a local variable
label_index=0; 
plt.style.use('ggplot');
fig=plt.figure() # Create a fig handler
ax=fig.add_subplot(111)

def sigmoid(z):
    """
    Computes the sigmoid function for a given matrix
    """
    g=1.0/(1.0+np.exp(-z))
    
    return g

def lrCostFunction(theta,X,y,lmbda=0.1,retGrad=False):
    """
    LRCOSTFUNCTION Compute cost and gradient for logistic regression with regularization
    J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
    theta as the parameter for regularized logistic regression and the
    radient of the cost w.r.t. to the parameters. 
    X=
    theta= number of features
    """
    # Initialize
    m=y.size # y is a column vector, so size = number of columns
    n=theta.size # theta is a column vector and size is number of rows
    # J=0
    
    # In mumpy * is the element-wise multiplication and .dot() is the matrix multiplication
    # Using the vector representation
    
    J=-(np.transpose(y).dot(np.log(sigmoid(X.dot(theta))))+np.transpose(1-y).dot(np.log(1-sigmoid(X.dot(theta)))))*1.0/m+ \
        (lmbda*1.0/(2*m))*np.sum(theta[1:n]*theta[1:n]); # In regularisation term, theta is from 1 to n (\theta_0 is not added)
    
    J_cost_all[label_index].append(J)
    
    # grad=np.zeros(n) # creates a column vector of size n 
   
    grad=(np.transpose(X).dot(sigmoid(X.dot(theta))-y))*1.0/m; # This element is present in all values of theta
    grad[1:n]=grad[1:n]+(lmbda*1.0/m)*theta[1:n]; # This adds the second term to all other than \theta_0 this is the regularization term
    
    # print(J)
    if retGrad:
        return J, grad
    else:
        return J
        
    #return J

def gradient_cost_funtion(theta,X,y,lmbda=0.1):
    
    n=theta.size # theta is a column vector and size is number of rows
    m=y.size # y is a column vector, so size = number of columns
    # grad=np.zeros(n) # creates a column vector of size n 
    
    grad=(np.transpose(X).dot(sigmoid(X.dot(theta))-y))*1.0/m; # This element is present in all values of theta
    grad[1:n]=grad[1:n]+(lmbda*1.0/m)*theta[1:n]; # This adds the second term to all other than \theta_0 this is the regularization term
    
    
    
    return grad


def oneVsAll(X,y,num_labels=10,lmbda=0.1,method='Newton-CG'):
    """
    ONEVSALL trains multiple logistic regression classifiers and returns all
    the classifiers in a matrix all_theta, where the i-th row of all_theta 
    corresponds to the classifier for label i
    """
    global label_index # The scope of the variable has to be declared here, o.w python thinks it is a local variable
    m=X.shape[0]; # Number of data elements (rows)
    n=X.shape[1]; # Number of features (\theta)
    
    all_theta=np.zeros((num_labels, n+1)) # n+1 because we have \theta_0 and number of labels are number of y's that you need to predict. Here it is 0-9
    
    # Make the X matrix columns same as theta ($theta_0,\ldots,\theta_n$)
    # Concate ones (\theta_0) to X
    
    X=np.concatenate((np.ones((m,1)),X),axis=1); # np.ones((m,1)) creates a 2d array
    
    initial_theta=np.zeros(n+1); # This is a 1d-array. Do not specify this as (n+1,1) as that would be a 2d array
    
        
    for c in range(0,num_labels):
        y_bool=(y==c).astype(int) # Create an 1d array
        # all_params=spo.fmin_cg(lrCostFunction,initial_theta,fprime=gradient_cost_funtion,args=(X,y_bool,lmbda),full_output=True,retall=1)
        # opt_theta=all_params[0]
        J_cost_all.append([]);
        all_params= spo.minimize(lrCostFunction, x0=initial_theta, args=(X,y_bool,lmbda,True), options={'disp': True,'maxiter':10000}, method=method, jac=True)
        
        # all_params = spo.minimize(lrCostFunction, x0=initial_theta, args=(X,y_bool,lmbda,True), options={'disp': True, 'maxiter':100}, method="SLSQP", jac=True)
        opt_theta=all_params.x
        
        # opt_theta=spo.minimize(lrCostFunction,initial_theta,jac=gradient_cost_funtion, args=(X,y_bool,lmbda),method='CG',options={'disp':True})# In this case opt_theta is a dictionary
        
        # all_theta[c,:]=opt_theta.x
        all_theta[c,:]=opt_theta
        print('Finished for the label {0}'.format(c))
        # Plot the cost function with respect to number of iterations
        
        ax.plot(J_cost_all[label_index],linewidth=1.5)
        plt.hold(True)
        label_index=label_index+1;
    
    ax.set_title('Evaluation of cost function using %s'%(method))
    ax.set_ylabel(r'Value of the cost function J($\theta$)')
    ax.set_xlabel('Number of gradient evaluations')
    ax.legend(['label-'+str(x) for x in range(0,10)],framealpha=0.8)
    plt.show()
    fig.savefig(os.path.join('plots','cost_vs_iterations_%s.eps'%method))
    return all_theta


def predictOneVsAll(all_theta,X):
    """
    PREDICT Predict the label for a trained one-vs-all classifier. The labels 
    are in the range 1..K, where K = size(all_theta, 1). 
    """
    m = X.shape[0];
    # num_labels = all_theta.shape[0];

    # You need to return the following variables correctly 
    # p = np.zeros(m);

    # Add ones to the X data matrix
    X=np.concatenate((np.ones((m,1)),X),axis=1);
    
    # Probability
    prob=sigmoid(X.dot(np.transpose(all_theta))); # This gives a vector of m x num_labels
    
    # Take the row-wise maximum and idex at which the maximum value occurs
    pred_digits=np.argmax(prob,axis=1)
    
    
    
    return pred_digits


if __name__=='__main__':
       
     
    from digit_recognizer import get_data
    import pandas as pd
    
   
    
    
    train_df=get_data();
    test_df=get_data(train=False)
    
    train_df=train_df.as_matrix()
    
    y=train_df[:,0]
  
    X=train_df[:,1:]
    
    X_test=test_df.as_matrix()
    
    
    outdir='results'
    try:
        os.makedirs(outdir)
    except:
        pass
    
    method='Newton-CG'
    opt_theta=oneVsAll(X,y,num_labels=10,lmbda=0.1,method=method)
    #TODO: You can save the opt_theta in a pickle form, so the optimization need not run every time
    
    np.savetxt('./%s/gd_opt_%s.csv'%(outdir,method),opt_theta,delimiter=',')
    
    train_pred_digits=predictOneVsAll(opt_theta,X)
    
    print("Training set accuracy: {0}".format(np.mean(train_pred_digits==y)*100))
    
    # Using the opt theta values
    
    test_pred_digits=predictOneVsAll(opt_theta, X_test)
    
    
    df_test_pred=pd.DataFrame(test_pred_digits,index=range(1,test_pred_digits.shape[0]+1),columns=['Label'])
    df_test_pred.index.names=['ImageId']
    
    df_test_pred.to_csv('%s/gd_test_results_%s.csv'%(outdir,method))
    
    
     
    