import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2
import itertools
import networkx as nx
import numpy as np
from networkx.algorithms.bipartite.matrix import biadjacency_matrix
from networkx.algorithms.bipartite import sets as bipartite_sets
import analyzers
import mltechs

# Load csv
def loadcsv(filename1, filename2) :
    train_data = pd.read_csv(filename1)
    train_data = train_data.to_numpy()
    test_data = pd.read_csv(filename2)
    test_data = test_data.to_numpy()
    return train_data, test_data

#Over sampling using SMOTE
def balance_usingSMOTE(X_train,Y_train):
    sm = SMOTE()
    X_train, Y_train = sm.fit_sample(X_train, Y_train)
    return X_train, Y_train


def chisquareFS(X_train,Y_train):
    num_feats = max((int)((X_train.shape[1]*15)/100),3)
    chi2_features = SelectKBest(chi2, k = num_feats)
    X_train = chi2_features.fit_transform(X_train, Y_train)
    return X_train


def minimum_weight_full_matching(G, top_nodes=None, weight='weight'):


    __all__ = ['maximum_matching', 'hopcroft_karp_matching', 'eppstein_matching',
               'to_vertex_cover', 'minimum_weight_full_matching']

    INFINITY = float('inf')

    try:
        import scipy.optimize
    except ImportError:
        raise ImportError('minimum_weight_full_matching requires SciPy: ' +
                          'https://scipy.org/')
    left, right = nx.bipartite.sets(G, top_nodes)
    # Ensure that the graph is complete. This is currently a requirement in
    # the underlying  optimization algorithm from SciPy, but the constraint
    # will be removed in SciPy 1.4.0, at which point it can also be removed
    # here.
    for (u, v) in itertools.product(left, right):
        # As the graph is undirected, make sure to check for edges in
        # both directions
        if (u, v) not in G.edges() and (v, u) not in G.edges():
            raise ValueError('The bipartite graph must be complete.')
    U = list(left)
    V = list(right)
    weights = biadjacency_matrix(G, row_order=U,
                                 column_order=V, weight=weight).toarray()
    left_matches = scipy.optimize.linear_sum_assignment(weights)
    d = {U[u]: V[v] for u, v in zip(*left_matches)}
    # d will contain the matching from edges in left to right; we need to
    # add the ones from right to left as well.
    d.update({v: u for u, v in d.items()})
    return d


def match_metrics(X_train,X_test,p,cutoff=0) :
    G = nx.Graph()
    n = X_train.shape[1]
    for i in range(X_train.shape[1]) :
        for j in range(X_test.shape[1]) :
            if (p[i][j] >= cutoff) :
                G.add_edge(i, n + j, weight=-p[i][j])
            else :
                G.add_edge(i, n + j, weight=100)
    y = minimum_weight_full_matching(G)

    X_test_new = []
    X_train_new = []
    v = 0
    h = 0
    z={}
    for i in range(X_train.shape[1]) :
        if i in y :
            h = h + p[i][y[i]-n]
    for i in range(X_train.shape[1]) :
        if i in y and p[i][y[i]-n] >= cutoff:
            v = v + p[i][y[i]-n]
            z[i] = y[i]
            X_test_new.append(X_test[:,y[i]-X_train.shape[1]])
            X_train_new.append(X_train[:,i])
    X_train = np.transpose(X_train_new)
    X_test = np.transpose(X_test_new)
  
    return X_train, X_test

def check(auc, x, fpr, tpr, pd, pf) :
    if x > auc :
        auc = x
        i1 = get_pdpf(fpr,tpr)
        pf = fpr[i1]
        pd = tpr[i1]
    return auc, pd, pf


def get_pdpf(fpr,tpr) :
    x = [(1 + fpr[i] - tpr[i]) for i in range(len(fpr))]
    return x.index(min(x))


def main_step(source_file,target_file,cutoff=0, n_iter = 100) :
    print(source_file,"--->",target_file)
    auc = 0
    pf = 0
    pd = 0
    source_data, target_data = loadcsv(source_file + ".csv",target_file + ".csv")

    X_source, Y_source = source_data[:,0:source_data.shape[1]-1],source_data[:,-1]
    X_target, Y_target = target_data[:,0:target_data.shape[1]-1],target_data[:,-1]

    X_source, Y_source = balance_usingSMOTE(X_source,Y_source)

    X_source = chisquareFS(X_source, Y_source)

    p = analyzers.KendalTau(X_source,X_target)

    X_source, X_target = match_metrics(X_source,X_target,p)
    if(len(X_source)== 0) :
        return -1;

    all_auc_tech = {}

    for i in range(n_iter):

        rf,Y_rf_test,Y_rf_train,rf_fpr,rf_tpr,rf_thresh = mltechs.RandomForest(X_source,Y_source,X_target,Y_target)
        nn,Y_nn_test,Y_nn_train,nn_fpr,nn_tpr,nn_thresh = mltechs.NN(X_source,Y_source,X_target,Y_target,100)
        knn,Y_knn_test,Y_knn_train,knn_fpr,knn_tpr,knn_thresh = mltechs.KNN(X_source,Y_source,X_target,Y_target)
        lr,Y_lr_test,Y_lr_train,lr_fpr,lr_tpr,lr_thresh = mltechs.Logistic_Regression(X_source,Y_source,X_target,Y_target)
        nb,Y_nb_test,Y_nb_train,nb_fpr,nb_tpr,nb_thresh = mltechs.NB(X_source,Y_source,X_target,Y_target)
        xgb,Y_xgb_test,Y_xgb_train,xgb_fpr,xgb_tpr,xgb_thresh = mltechs.XGBoosting(X_source,Y_source,X_target,Y_target)
        cnn,Y_cnn_test,Y_cnn_train,cnn_fpr,cnn_tpr,cnn_thresh = mltechs.CNN_1D(X_source,Y_source,X_target,Y_target)
        svm, Y_svm_test, Y_svm_train, svm_fpr, svm_tpr, svm_thresh = mltechs.SVM(X_source,Y_source,X_target,Y_target)



        auc, pd, pf = check(auc, cnn, cnn_fpr, cnn_tpr, pd, pf)
        auc, pd, pf = check(auc, knn, knn_fpr, knn_tpr, pd, pf)
        auc, pd, pf = check(auc, nn, nn_fpr, nn_tpr, pd, pf)
        auc, pd, pf = check(auc, rf, rf_fpr, rf_tpr, pd, pf)
        auc, pd, pf = check(auc, nb, nb_fpr, nb_tpr, pd, pf)
        auc, pd, pf = check(auc, lr, lr_fpr, lr_tpr, pd, pf)
        auc, pd, pf = check(auc, xgb, xgb_fpr,xgb_tpr, pd, pf)
        auc, pd, pf = check(auc, svm, svm_fpr,svm_tpr, pd, pf)


        all_auc_tech[i] = (auc, pf, pd)

    return all_auc_tech




def generateResults(all_auc, n_iter=100) :
    import statistics 
    col_names = ['Target','AUC','Pf','Pd']
    u = []
    for key in all_auc:
        y = []
        z = []

        for k2 in all_auc[key]:
            h5 = []
            h3 = []
            for i in range(n_iter):
                h4 = []
                if i in all_auc[key][k2]:
                  h4.append(all_auc[key][k2][i][0])
                  h4.append(all_auc[key][k2][i][1])
                  h4.append(all_auc[key][k2][i][2])
                  h5.append(h4)

            if(len(h5) == 0) :
              continue;
            h5 = sorted(h5,key=lambda x: x[0])

            if(len(h5)%2) == 0 :
                ind = int(len(h5)/2)
                h3.append((h5[ind][0] + h5[ind-1][0])/2)
                h3.append((h5[ind][1] + h5[ind-1][1])/2)
                h3.append((h5[ind][2] + h5[ind-1][2])/2)
            else :
                ind = int(len(h5)/2)
                h3.append(h5[ind][0])
                h3.append(h5[ind][1])
                h3.append(h5[ind][2])

            print(k2 + "->" + key)
            z.append(h3)
        z = sorted(z,key=lambda x: x[0])
#         print(z)
        y.append(key)
        if(len(z)%2) == 0 :
            ind = int(len(z)/2)
            y.append((z[ind][0] + z[ind-1][0])/2)
            y.append((z[ind][1] + z[ind-1][1])/2)
            y.append((z[ind][2] + z[ind-1][2])/2)
        else :
            ind = int(len(z)/2)
            y.append(z[ind][0])
            y.append(z[ind][1])
            y.append(z[ind][2])
        u.append(y)
    my_df  = pd.DataFrame(u,columns = col_names)
    fileName = 'Results.xlsx'
    my_df.to_excel(fileName)