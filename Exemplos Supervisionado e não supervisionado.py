# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:47:45 2017

@author: R337555
"""

# Exemplo Supervivionado - Não supervisionado

from sklearn import datasets
digits = datasets.load_digits()
X, y = digits.data, digits.target

print (X.shape)
print (y.shape)

print( X[666].reshape((8,8)))
from skimage import io as io
io.imshow(X[0].reshape((8,8)))
io.show()

#Visualize some of the data.
import matplotlib.pyplot as plt
fig, ax = plt.subplots(4, 4, subplot_kw={'xticks':[], 'yticks':[]})
for i in range(ax.size):
    ax.flat[i].imshow(digits.data[i].reshape(8, 8), cmap=plt.cm.Blues)
    
    
# Scikit-learn
# these two lines are here just to hide python warnings. Ignore them
import warnings
warnings.filterwarnings("ignore")

from sklearn import neighbors
knn = neighbors.KNeighborsClassifier()
knn.fit(X,y) 
y_pred = knn.predict(X)
for i in range(16):
    print (i, ": ", y_pred[i])
    
 knn.score(X,y)
 
 #Como medir a eficácia do algoritmo
 import pylab as plt
from sklearn import metrics

def plot_confusion_matrix(y, y_pred):
    plt.imshow(metrics.confusion_matrix(y, y_pred),
               cmap=plt.cm.jet, interpolation='nearest')
    plt.colorbar()
    plt.ylabel('true value')
    plt.xlabel('predicted value')
    
print ("Success rate:", metrics.accuracy_score(y, y_pred))
plot_confusion_matrix(y, y_pred)
print (metrics.classification_report(y_pred,y))
 
 
 import numpy as np

digits = datasets.load_digits()
X, y = digits.data, digits.target

perm = np.random.permutation(y.size)

splitting = 0.7
split_point = int(np.ceil(y.shape[0]*splitting))

X_train = X[perm[:split_point].ravel(),:]
y_train = y[perm[:split_point].ravel()]

X_test = X[perm[split_point:].ravel(),:]
y_test = y[perm[split_point:].ravel()]

#Train a classifier on training data
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

#Check on the test set and visualize performance
yhat=knn.predict(X_test)
print ("Success rate: ", metrics.accuracy_score(yhat, y_test))
plt.figure()
plt.imshow(metrics.confusion_matrix(y_test, yhat), interpolation='nearest')
plt.colorbar()

from sklearn.cross_validation import train_test_split
from sklearn import metrics

splitting = 0.7
acc=np.zeros((10,))
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=splitting)
    knn = neighbors.KNeighborsClassifier()
    knn.fit(X_train,y_train)
    yhat=knn.predict(X_test)
    acc[i] = metrics.accuracy_score(yhat, y_test)
print ("Score average: " + str(np.mean(acc[0])))

# Seleção do melhor modelo
# Validação cruzada (Cross-Validation)

from sklearn.cross_validation import train_test_split
from sklearn import neighbors
from sklearn import tree
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

digits = datasets.load_digits()
X, y = digits.data, digits.target

splitting = 0.7
acc_cr=np.zeros((10,3))
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=splitting)
    knn = neighbors.KNeighborsClassifier()
    svc = svm.SVC()
    dt = tree.DecisionTreeClassifier()
    
    knn.fit(X_train,y_train)
    svc.fit(X_train,y_train)
    dt.fit(X_train,y_train)
    
    yhat_knn=knn.predict(X_test)
    yhat_svc=svc.predict(X_test)
    yhat_dt=dt.predict(X_test)
    
    acc_cr[i][0] = metrics.accuracy_score(yhat_knn, y_test)
    acc_cr[i][1] = metrics.accuracy_score(yhat_svc, y_test)
    acc_cr[i][2] = metrics.accuracy_score(yhat_dt, y_test)


plt.boxplot(acc_cr);

ax = plt.gca()
ax.set_xticklabels(['KNN','SVM','Trees'])


# Validação cruzada de K iterações (K-fold cross validation)
from sklearn import cross_validation
acc_kf = np.zeros((10,3))
kf=cross_validation.KFold(n=y.shape[0], n_folds=10, shuffle=True, random_state=0)
i=0
for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    knn = neighbors.KNeighborsClassifier()
    svc = svm.SVC()
    dt = tree.DecisionTreeClassifier()
    
    knn.fit(X_train,y_train)
    svc.fit(X_train,y_train)
    dt.fit(X_train,y_train)
    
    yhat_knn=knn.predict(X_test)
    yhat_svc=svc.predict(X_test)
    yhat_dt=dt.predict(X_test)
    
    acc_kf[i][0] = metrics.accuracy_score(yhat_knn, y_test)
    acc_kf[i][1] = metrics.accuracy_score(yhat_svc, y_test)
    acc_kf[i][2] = metrics.accuracy_score(yhat_dt, y_test)
    i=i+1
    

plt.boxplot(acc_kf);
ax = plt.gca()
ax.set_xticklabels(['KNN','SVM','Trees'])

# Validação cruzada deixando um fora (Leave-one-out cross-validation)
X_reduced=X[:300]
y_reduced=y[:300]


from sklearn import cross_validation
acc_loo = np.zeros((y_reduced.shape[0],3))
loo = cross_validation.LeaveOneOut(n=y_reduced.shape[0])
i=0
for train_index, test_index in loo:
    X_train, X_test = X_reduced[train_index], X_reduced[test_index]
    y_train, y_test = y_reduced[train_index], y_reduced[test_index]
    
    knn = neighbors.KNeighborsClassifier()
    svc = svm.SVC()
    dt = tree.DecisionTreeClassifier()
    
    knn.fit(X_train,y_train)
    svc.fit(X_train,y_train)
    dt.fit(X_train,y_train)
    
    yhat_knn=knn.predict(X_test)
    yhat_svc=svc.predict(X_test)
    yhat_dt=dt.predict(X_test)
    
    acc_loo[i][0] = metrics.accuracy_score(yhat_knn, y_test)
    acc_loo[i][1] = metrics.accuracy_score(yhat_svc, y_test)
    acc_loo[i][2] = metrics.accuracy_score(yhat_dt, y_test)
    i=i+1
    
    
print ("Success rate: \n", \
"KNN: ", (sum(acc_loo)/acc_loo.shape[0])[0], "\n",\
"SVM: ", (sum(acc_loo)/acc_loo.shape[0])[1],  "\n",\
"Tree: ", (sum(acc_loo)/acc_loo.shape[0])[2])

plt.boxplot(acc_loo);
ax = plt.gca()
ax.set_ylim(-3, 3)
ax.set_xticklabels(['KNN','SVM','Tree'])


### Alguns modelos supervisionados (predizendo churn)
from __future__ import division
import pandas as pd
import numpy as np

churn_df = pd.read_csv('D:/DatAcademy/Mod_06. Data Science com Phyton. Ed 3/churn.csv')
col_names = churn_df.columns.tolist()

print( "Column names:")
print (col_names)

churn_df.head(6)

churn_result = churn_df['Churn?']
y = np.where(churn_result == 'True.',1,0)
print ("Churns positive: ", sum(y))
print ("Churns negative: ", y.shape[0] - sum(y))

# We don't need these columns
to_drop = ['State','Phone','Churn?']
churn_feat_space = churn_df.drop(to_drop,axis=1)

# 'yes'/'no' has to be converted to boolean values
# NumPy converts these from boolean to 1. and 0. later
yes_no_cols = ["Int'l Plan","VMail Plan"]
churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'

# Pull out features for future use
features = churn_feat_space.columns

X = churn_feat_space.as_matrix().astype(np.float)

print ("There are %d examples and %d variables" % X.shape)

# Vizinho mais próximo (nearest neighbour)
from sklearn import cross_validation
from sklearn import neighbors
from sklearn import metrics
acc = np.zeros((5,))
i=0
kf=cross_validation.KFold(n=y.shape[0], n_folds=5, shuffle=False, random_state=0)
#We will build the predicted y from the partial predictions on the test of each of the folds
yhat = y.copy()
for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    dt = neighbors.KNeighborsClassifier(n_neighbors=3)
    dt.fit(X_train,y_train)
    yhat[test_index] = dt.predict(X_test)
    acc[i] = metrics.accuracy_score(yhat[test_index], y_test)
    i=i+1
print ('Mean accuracy: '+ str(np.mean(acc)))

cm = metrics.confusion_matrix(y, yhat)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.matshow(cm)
plt.title('Confussion matrix',size=20)
ax.set_xticklabels([''] + ['no churn', 'churn'], size=20)
ax.set_yticklabels([''] + ['no churn', 'churn'], size=20)
plt.ylabel('Predicted',size=20)
plt.xlabel('Actual',size=20)
for i in range(2):
    for j in range(2):
        ax.text(i, j, cm[i,j], va='center', ha='center',color='white',size=20)
fig.set_size_inches(7,7)
plt.show()

print (metrics.classification_report(y,yhat))

# Standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

acc = np.zeros((5,))
i=0
yhat = y.copy()
for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    dt = neighbors.KNeighborsClassifier(3)
    dt.fit(X_train,y_train)
    X_test = scaler.transform(X_test)
    yhat[test_index] = dt.predict(X_test)
    acc[i] = metrics.accuracy_score(yhat[test_index], y_test)
    i=i+1
print ('Average of scores: '+ str(np.mean(acc)))

cm = metrics.confusion_matrix(y, yhat)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.matshow(cm)
plt.title('Confusion matrix',size=20)
ax.set_xticklabels([''] + ['no churn', 'churn'], size=20)
ax.set_yticklabels([''] + ['no churn', 'churn'], size=20)
plt.ylabel('Predicted',size=20)
plt.xlabel('Actual',size=20)
for i in range(2):
    for j in range(2):
        ax.text(i, j, cm[i,j], va='center', ha='center',color='white',size=20)
fig.set_size_inches(7,7)
plt.show()
print (metrics.classification_report(y,yhat))


 # Árvores de decisão
 import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn import metrics

churn_df = pd.read_csv('churn.csv')
col_names = churn_df.columns.tolist()
churn_result = churn_df['Churn?']
y = np.where(churn_result == 'True.',1,0)
to_drop = ['State','Phone','Churn?']
churn_feat_space = churn_df.drop(to_drop,axis=1)
yes_no_cols = ["Int'l Plan","VMail Plan"]
churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'
features = churn_feat_space.columns
X = churn_feat_space.as_matrix().astype(np.float)

# up to here same as before: load and clean

kf=cross_validation.KFold(n=y.shape[0], n_folds=5, shuffle=False, random_state=0)

acc = np.zeros((5,))
i=0
#We will build the predicted y from the partial predictions on the test of each of the folds
yhat = y.copy()
for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    dt = tree.DecisionTreeClassifier(criterion='entropy')
    dt.fit(X_train,y_train)
    X_test = scaler.transform(X_test)
    yhat[test_index] = dt.predict(X_test)
    acc[i] = metrics.accuracy_score(yhat[test_index], y_test)
    i=i+1
print( 'Score average: '+ str(np.mean(acc)))

cm = metrics.confusion_matrix(y, yhat)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.matshow(cm)
plt.title('Confusion matrix',size=20)
ax.set_xticklabels([''] + ['no churn', 'churn'], size=20)
ax.set_yticklabels([''] + ['no churn', 'churn'], size=20)
plt.ylabel('Predicted',size=20)
plt.xlabel('Actual',size=20)
for i in range(2):
    for j in range(2):
        ax.text(i, j, cm[i,j], va='center', ha='center',color='white',size=20)
fig.set_size_inches(7,7)
plt.show()

print (metrics.classification_report(y,yhat))


from sklearn import tree
import os
with open('tree.dot', 'w') as dotfile:
    tree.export_graphviz(
        dt,
        dotfile)
    
os.system("dot -Tpng tree.dot -o tree.png")
from IPython.core.display import Image
Image("tree.png")

# Os resultados são melhores que no caso de KNN. Podemos ver alguns parâmetros da árvore.

import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import metrics

churn_df = pd.read_csv('churn.csv')
col_names = churn_df.columns.tolist()
churn_result = churn_df['Churn?']
y = np.where(churn_result == 'True.',1,0)
to_drop = ['State','Phone','Churn?']
churn_feat_space = churn_df.drop(to_drop,axis=1)
yes_no_cols = ["Int'l Plan","VMail Plan"]
churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'
features = churn_feat_space.columns
X = churn_feat_space.as_matrix().astype(np.float)

# up to here same as before: load and clean

kf=cross_validation.KFold(n=y.shape[0], n_folds=5, shuffle=False, random_state=0)

acc = np.zeros((5,))
i=0
#We will build the predicted y from the partial predictions on the test of each of the folds
yhat = y.copy()
for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    #Standard parameters
    clf = svm.SVC(kernel='rbf', gamma = 0.051, C = 1)
    clf.fit(X_train,y_train.ravel())
    X_test = scaler.transform(X_test)
    yhat[test_index] = clf.predict(X_test)
    acc[i] = metrics.accuracy_score(yhat[test_index], y_test)
    i=i+1
print ('Score average: '+ str(np.mean(acc)))

cm = metrics.confusion_matrix(y, yhat)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.matshow(cm)
plt.title('Confusion matrix',size=20)
ax.set_xticklabels([''] + ['no churn', 'churn'], size=20)
ax.set_yticklabels([''] + ['no churn', 'churn'], size=20)
plt.ylabel('Predicted',size=20)
plt.xlabel('Actual',size=20)
for i in range(2):
    for j in range(2):
        ax.text(i, j, cm[i,j], va='center', ha='center',color='white',size=20)
fig.set_size_inches(7,7)
plt.show()

print (metrics.classification_report(y,yhat))

# Naive-Bayes: Classificador probabilístico baseado na aplicação do teorema de Bayes
import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt

churn_df = pd.read_csv('churn.csv')
col_names = churn_df.columns.tolist()
churn_result = churn_df['Churn?']
y = np.where(churn_result == 'True.',1,0)
to_drop = ['State','Phone','Churn?']
churn_feat_space = churn_df.drop(to_drop,axis=1)
yes_no_cols = ["Int'l Plan","VMail Plan"]
churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'
features = churn_feat_space.columns
X = churn_feat_space.as_matrix().astype(np.float)

# up to here same as before: load and clean

kf=cross_validation.KFold(n=y.shape[0], n_folds=5, shuffle=False, random_state=0)

acc = np.zeros((5,))
i=0
#We will build the predicted y from the partial predictions on the test of each of the folds
yhat = y.copy()
for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    #Standard parameters
    clf = GaussianNB()
    clf.fit(X_train,y_train.ravel())
    X_test = scaler.transform(X_test)
    yhat[test_index] = clf.predict(X_test)
    acc[i] = metrics.accuracy_score(yhat[test_index], y_test)
    i=i+1
print ('Score average: '+ str(np.mean(acc)))

cm = metrics.confusion_matrix(y, yhat)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.matshow(cm)
plt.title('Confusion matrix',size=20)
ax.set_xticklabels([''] + ['no churn', 'churn'], size=20)
ax.set_yticklabels([''] + ['no churn', 'churn'], size=20)
plt.ylabel('Predicted',size=20)
plt.xlabel('Actual',size=20)
for i in range(2):
    for j in range(2):
        ax.text(i, j, cm[i,j], va='center', ha='center',color='white',size=20)
fig.set_size_inches(7,7)
plt.show()

print (metrics.classification_report(y,yhat))








 
 
 