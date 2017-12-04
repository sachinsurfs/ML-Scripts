import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_hastie_10_2
import matplotlib.pyplot as plt
import math
import csv
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

def get_error_rate(pred, Y):
    y_len = float(len(Y))
    err  = 0.0
    for i in range(len(Y)):
        err += int(int(pred[i]) != int(Y[i]))

    return  100*(err/y_len)   
    #return sum(int(pred != Y)) / y_len

def generic_clf(Y_train, X_train, Y_test, X_test, clf):
    clf.fit(X_train,Y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)
    
def margin_clf(Y_train, X_train, Y_test, X_test, M, clf, rho=0):
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
    
    #Create Model
    for i in range(M):
        # Train base classifier with W
        clf.fit(X_train, Y_train, sample_weight = w)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)

        #h(x) expressed in terms of {1,-1}
        pred_train_i1 = [1 if x=='1' else -1 for x in pred_train_i]
        pred_test_i1 = [1 if x=='1' else -1 for x in pred_test_i]

        # Indicator function
        In = [int(x) for x in (pred_test_i != Y_test)]
         
        #h(x)y in terms of {1,-1}
        hxy = [x if x==1 else -1 for x in In]
        

        # Error
        err_m = np.dot(w,In) / sum(w)
        #err_m = float(sum(In)) / n_train

        err_thr = (1.0-rho)/2.0
        if(err_m >= err_thr) :
            #print err_m
            continue

        #print err_m
        # Alpha
        alpha_m = 0.5 * np.log(((1.0-rho) *(1.0 - err_m) )/ ((1.0+rho)*err_m))

        #print alpha_m
        #Normalization Factor 
        Z_t = 2*( math.sqrt((err_m*(1-err_m)) / (1-rho**2) ) )
        
        #w, np.exp([float(x) * alpha_m for x in In])

        # Update Distribution        
        w = np.divide(np.multiply(w, np.exp([float(x) * alpha_m for x in hxy])), Z_t)

        #print sum(w)
        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train, 
                                          [float(x) * alpha_m for x in pred_train_i1])]
        pred_test = [sum(x) for x in zip(pred_test, 
                                         [float(x) * alpha_m for x in pred_test_i1])]
    

    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    
    pred_train = [x if x==1 else 0 for x in pred_train]
    pred_test = [x if x==1 else 0 for x in pred_test]
    # Return error rate in train and test set
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)


if __name__ == '__main__':
    

    reader = csv.reader(open("spambase.data.shuffled", "rb"), delimiter=",")
    data = list(reader)

    m = 3000

    T = {100, 200, 500, 1000}

    rho = { 0.0009765625,0.001953125,0.00390625,0.0078125, 0.015625,0.03125,0.0625, 0.125, 0.25, 0.5 }
   
    X_train = []
    Y_train = []

    t_cross_error = []
    t_fit_error  = []

    training = data[:m]
    testing = data[m:]

    D = [[]] * m

    for i in range(len(training)):
        X_train.append(list(training[i][:-1]))
        Y_train.append(training[i][-1])

    X_test = []
    Y_test = []

    for i in range(len(testing)):
        X_test.append(list(testing[i][:-1]))
        Y_test.append(testing[i][-1])


    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)   

    
    # Fit a simple decision tree first
    clf_tree = DecisionTreeClassifier(max_depth = 1, random_state = 1)
    er_tree = generic_clf(Y_train, X_train, Y_test, X_test, clf_tree)
    
    # Fit Adaboost classifier using a decision tree as base estimator
    # Test with different number of iterations
#    er_train, er_test = [er_tree[0]], [er_tree[1]]

    for j in rho: 
        er_train, er_test = [], []
        for i in T:   
            er_i = margin_clf(Y_train, X_train, Y_test, X_test, i, clf_tree,j)
            er_train.append(er_i[0])
            er_test.append(er_i[1])
        print "T no :"+str(i)+" rho no : "+str(j)
        print er_train
        print er_test
    #print er_train
    #print er_test
    # Compare error rate vs number of iterations
       # print_error_rate(er_train)
