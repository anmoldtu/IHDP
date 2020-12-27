from scipy import stats
import numpy as np
import pandas as pd
import random


#PAnalyzer
def PAnalyzer(X_train,X_test) :
    p = [[0 for j in range(X_test.shape[1])] for i in range(X_train.shape[1])]
    for i in range(X_train.shape[1]) :
        for j in range(X_test.shape[1]) :
            for k in range(10,100,10) :
                sm = np.percentile(X_train[:,i], k)
                bg = np.percentile(X_test[:,j], k)
                if(sm > bg) :
                    sm,bg = bg,sm
                if(bg != 0) :
                    sve = sm/bg
                else :
                    sve = 0
                p[i][j] = p[i][j] + sve
            p[i][j] = p[i][j]/9.0
    return p


#KSAnalyzer
def KAnalyzer(X_train,X_test) :
    pk = [[0 for j in range(X_test.shape[1])] for i in range(X_train.shape[1])]
    for i in range(X_train.shape[1]) :
        for j in range(X_test.shape[1]) :
            x, pk[i][j] = stats.ks_2samp(X_train[:,i], X_test[:,j])
    return pk


#Kendal Tau Correlation Coefficient
def KendalTau(X_train,X_test) :
    k = [[0 for j in range(X_test.shape[1])] for i in range(X_train.shape[1])]
    sm = X_train.shape[0]
    bg = X_test.shape[0]
    n_iter = 75
    if(sm > bg) :
        sm,bg = bg,sm
    for w in range(n_iter):
        random.seed(w)
        randomRow = random.sample(range(0, bg),sm)
        list.sort(randomRow)
    #     print(randomRow)
        if(X_train.shape[0] == bg) :
            X_train_1 = X_train[randomRow,:]
        else :
            X_train_1 = X_train
        if(X_test.shape[0] == bg) :
            X_test_1 = X_test[randomRow,:]
        else :
            X_test_1 = X_test
        for i in range(X_train.shape[1]) :
            for j in range(X_test.shape[1]) :
                u,x = stats.kendalltau(X_train_1[:,i], X_test_1[:,j],nan_policy = 'omit')
                k[i][j] = k[i][j] + u
    for i in range(X_train.shape[1]) :
            for j in range(X_test.shape[1]) :
                k[i][j] = k[i][j]/n_iter
#     print(k)
    k = np.asarray(k)
    k = np.nan_to_num(k,0)
    return k