import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.framework import ops
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def RandomForest(X_train, Y_train, X_test, Y_test) :
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth = 5, random_state=0)
    clf.fit(X_train, Y_train)
    Y_pred_test=clf.predict(X_test)
    Y_pred_train=clf.predict(X_train)

    Y_pred_probs = clf.predict_proba(X_test)
    Y_pred_probs = Y_pred_probs[:,1]

    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_probs)
    
    return metrics.roc_auc_score(Y_test,Y_pred_probs),Y_pred_test,Y_pred_train,fpr,tpr,thresholds


#SVM_LinearSVC
def SVM(X_train, Y_train, X_test, Y_test) :
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    svm = LinearSVC()
    clf = CalibratedClassifierCV(svm) 
    clf.fit(X_train,Y_train)
    Y_pred_test = clf.predict(X_test)
    Y_pred_train=clf.predict(X_train)
    Y_pred_probs = clf.predict_proba(X_test)
    Y_pred_probs = Y_pred_probs[:,1]
#     generate_roc_curve(Y_test,Y_pred_test)
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_probs)
    return metrics.roc_auc_score(Y_test,Y_pred_probs),Y_pred_test,Y_pred_train,fpr,tpr,thresholds


#1D-CNN

def make_feature_count_nine(data):
  x = data.shape[1]
  if x==9:
    return data
  y = data.shape[0]
  b = np.zeros((y,9-x))
  adjusted_data = np.hstack((data,b))
  return adjusted_data
# For making sure that input data has 9 columns


def CNN_1D(X_train, Y_train, X_test, Y_test, EPOCHS = 50) :

  from tensorflow.keras import datasets, layers, models
  import matplotlib.pyplot as plt

  X_train = make_feature_count_nine(X_train)
  X_test = make_feature_count_nine(X_test)
  X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
  X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

  print(X_train.shape)
  print(X_test.shape)

  model = models.Sequential()
  model.add(layers.Conv1D(filters=4,kernel_size=2,strides=1,padding='same',activation='relu',input_shape=(9,1)))
  model.add(layers.AveragePooling1D())
  model.add(layers.Conv1D(8,2,activation='relu'))
  model.add(layers.Flatten())
  model.add(layers.Dense(1,activation='sigmoid'))
  model.compile(loss = 'binary_crossentropy',optimizer = "adam",metrics = [tf.keras.metrics.AUC()])
  # model.summary()

  history = model.fit(X_train, Y_train, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0)
  loss, auc = model.evaluate(X_test,Y_test, verbose=0)
  # print("Testing set AUC: {:5.2f} ".format(auc))

  Y_pred_probs = model.predict_proba(X_test)
  Y_pred_train=model.predict(X_train).flatten()
  Y_pred_test=model.predict(X_test).flatten() 
  fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_probs)
  # generate_roc_curve(Y_test,Y_pred)
  return auc,Y_pred_test,Y_pred_train,fpr,tpr,thresholds





class PrintDot(keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: 
            print('')
        print('.', end='')





#Neural Network Function
def NN(X_train, Y_train, X_test, Y_test,EPOCHS) :
    model = keras.Sequential([
        layers.Dense(18,kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', input_shape=[X_train.shape[1]]),
        layers.Dense(15,kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'),
        layers.Dense(10,kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'),
        layers.Dense(5,kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'),
        layers.Dense(1,activation='sigmoid')
     ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss=tf.keras.losses.Poisson(),
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.AUC()])
    
    history = model.fit(X_train, Y_train, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0)
    loss, auc = model.evaluate(X_test,Y_test, verbose=0)
#     print("Testing set AUC: {:5.2f} ".format(auc))
    Y_pred_test=model.predict(X_test).flatten()
    
    Y_pred_train=model.predict(X_train).flatten()
    
    Y_pred_probs = model.predict_proba(X_test)
#     print(Y_pred_probs.shape)
#     Y_pred_probs = Y_pred_probs[:,1]
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_probs)
#     auc_test = metrics.roc_auc_score(Y_test,Y_pred_probs)
#     print(auc_test)
    
#     generate_roc_curve(Y_test,Y_pred)
    return auc,Y_pred_test,Y_pred_train,fpr,tpr,thresholds


def NN_ensemble(X_train, Y_train, X_test, Y_test,EPOCHS) :
    model = keras.Sequential([
        layers.Dense(5,kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', input_shape=[X_train.shape[1]]),
        layers.Dense(2,kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'),
        layers.Dense(1,activation='sigmoid')
     ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss=tf.keras.losses.Poisson(),
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.AUC()])
    
    history = model.fit(X_train, Y_train, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0)
    loss, auc = model.evaluate(X_test,Y_test, verbose=0)
    
#     print("Testing set AUC: {:5.2f} ".format(auc))
#     Y_pred=model.predict(X_test).flatten()
    Y_pred_probs = model.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_probs)
#     generate_roc_curve(Y_test,Y_pred)
    return auc, fpr, tpr, thresholds




# K- nearest neighbors
def KNN(X_train,Y_train,X_test,Y_test) :
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier()
    neigh.fit(X_train, Y_train) 
    Y_pred_test=neigh.predict(X_test)
    Y_pred_train=neigh.predict(X_train)
#     generate_roc_curve(Y_test,Y_pred)
    Y_pred_probs = neigh.predict_proba(X_test)
#     generate_roc_curve(Y_test,Y_pred_test)
    Y_pred_probs = Y_pred_probs[:,1]
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_probs)
    return metrics.roc_auc_score(Y_test,Y_pred_probs),Y_pred_test,Y_pred_train,fpr,tpr,thresholds




# Naive Bayes
def NB(X_train,Y_train,X_test,Y_test) :
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(X_train,Y_train)
    Y_pred_test = clf.predict(X_test)
    Y_pred_train = clf.predict(X_train)
    Y_pred_probs = clf.predict_proba(X_test)
#     generate_roc_curve(Y_test,Y_pred_test)

    Y_pred_probs = Y_pred_probs[:,1]
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_probs)
    return metrics.roc_auc_score(Y_test,Y_pred_probs),Y_pred_test,Y_pred_train,fpr,tpr,thresholds



# Logistic Regression
def Logistic_Regression(X_train,Y_train,X_test,Y_test) :
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit(X_train,Y_train)
    Y_pred_test = clf.predict(X_test)
    Y_pred_train = clf.predict(X_train)
    Y_pred_probs = clf.predict_proba(X_test)
    Y_pred_probs = Y_pred_probs[:,1]
#     generate_roc_curve(Y_test,Y_pred_test)

    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_probs)
    return metrics.roc_auc_score(Y_test,Y_pred_probs),Y_pred_test,Y_pred_train,fpr,tpr,thresholds



# XGBoost
def XGBoosting(X_train,Y_train,X_test,Y_test) :
    from xgboost import XGBClassifier
    clf = XGBClassifier()
    clf.fit(X_train,Y_train)
    Y_pred_test = clf.predict(X_test)
    Y_pred_train = clf.predict(X_train)
    Y_pred_probs = clf.predict_proba(X_test)
    Y_pred_probs = Y_pred_probs[:,1]
#     generate_roc_curve(Y_test,Y_pred_test)
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_probs)
    return metrics.roc_auc_score(Y_test,Y_pred_probs),Y_pred_test,Y_pred_train,fpr,tpr,thresholds