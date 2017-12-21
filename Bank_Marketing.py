import numpy, os, csv
import pandas as pd
import sklearn.neural_network as nn
import matplotlib.pyplot as plt
import sklearn.metrics  as met
from sklearn.tree import DecisionTreeClassifier
from graphviz import Source
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing, tree
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import StratifiedKFold

# Read data file
data = pd.read_csv("bank-additional-full.csv", sep=";")

# Data Preprocessing
# Removing NaN values
def nan_format(x):
    if x == 'unknown':
        return float('NAN')
    else:
        return x

data['education'] = data['education'].apply(nan_format)
data['housing'] = data['housing'].apply(nan_format)
data['loan'] = data['loan'].apply(nan_format)
data['default'] = data['default'].apply(nan_format)
data['marital'] = data['marital'].apply(nan_format)
data['job'] = data['job'].apply(nan_format)

data=data.dropna()

# Dropping trivial data attributes
data = data.drop(['duration','poutcome','pdays','contact'],axis=1)

# Converting continuous attributes to categorical values [age]
data['age'] = pd.cut(data['age'],bins=4,labels=False)

# Encoding labels as numeric
categorical_cols = ['job','marital','education','default','housing','loan','month','day_of_week','y']

le = preprocessing.LabelEncoder()
for c in categorical_cols:
	if c in data:
		le.fit(data[c])
		data[c] = le.transform(data[c])

# Splitting test and train data
X = data.drop('y',axis=1).as_matrix()
Y = data['y'].as_matrix()
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, Y, stratify=data['y'], random_state=7)

"""
acc = 0
skf = StratifiedKFold(n_splits=100)
for train, test in skf.split(X, Y):
    X_train_d=X[train]
    X_test_d=X[test]
    y_train_d=Y[train]
    y_test_d=Y[test]
    
    """

# Fitting decision tree
d_tree = DecisionTreeClassifier(random_state=7, criterion='entropy', max_leaf_nodes = 18)
d_tree.fit(X_train_d, y_train_d)

# Calculating confusion matrix - TPR and FPR
y_pred_d = d_tree.predict(X_test_d)

dtree_accuracy=round(d_tree.score(X_test_d, y_test_d),5)
#acc += dtree_accuracy
dtree_prf_support = precision_recall_fscore_support(y_test_d, y_pred_d, average = "binary")
cf_matrix = confusion_matrix(y_test_d, y_pred_d)

print("Decision Tree:")
print("Accuracy: " , dtree_accuracy, ", Precision: ", round(dtree_prf_support[0],5), ", Recall: ", round(dtree_prf_support[1],5), ", F-Score: ", round(dtree_prf_support[2],5))
print("Confusion matrix:")
print(cf_matrix)

#print("Accuracy for decision tree after cross validation is: ",acc/100)

# Visualizing decision tree using model
feat_names = list(data.drop('y',axis=1).columns.values)
tree.export_graphviz(d_tree,out_file='d_tree.dot', feature_names = feat_names)
with open('d_tree.dot', 'r') as content_file:
	temp = content_file.read()
s = Source(temp, filename="tree_img", format="png")
s.view()
os.remove('tree_img')
os.remove('d_tree.dot')


# One hot encoding
nb_data=pd.get_dummies(data, columns=categorical_cols[:-1])
nb_data=data[:]
# Standardize continuous values
nb_data[["age","cons.price.idx","euribor3m","campaign","previous","nr.employed"]]/=nb_data[["age","cons.price.idx","euribor3m","campaign","previous","nr.employed"]].max()
nb_data["emp.var.rate"]=preprocessing.scale(nb_data["emp.var.rate"])
nb_data["cons.conf.idx"]=preprocessing.scale(nb_data["cons.conf.idx"])

# Splitting test and train data
X = nb_data.drop('y',axis=1).as_matrix()
Y = nb_data['y'].as_matrix()
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X, Y, stratify=data['y'], random_state=7)

"""
acc = 0
for train, test in skf.split(X, Y):
    X_train_n=X[train]
    X_test_n=X[test]
    y_train_n=Y[train]
    y_test_n=Y[test]
"""

# Fitting guassian naive bayes
naivebayes = GaussianNB()
naivebayes.fit(X_train_n, y_train_n)
naivebayes_pred = naivebayes.predict(X_test_n)

naive_accuracy = round(accuracy_score(y_test_n, naivebayes_pred, normalize = True),5)
#acc += naive_accuracy
naive_prf_support = precision_recall_fscore_support(y_test_n, naivebayes_pred, average = "binary")
cf_matrix=[]
cf_matrix = confusion_matrix(y_test_n, naivebayes_pred)

print("Naive Bayes:")
print("Accuracy: " , naive_accuracy, ", Precision: ", round(naive_prf_support[0],5), ", Recall: ", round(naive_prf_support[1],5), ", F-Score: ", round(naive_prf_support[2],5))
print("Confusion matrix:")
print(cf_matrix)

#print("Accuracy for Naive tree after cross validation: ",acc/100)


# Splitting test and train data
X = data.drop('y',axis=1).as_matrix()
Y = data['y'].as_matrix()
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, Y, stratify=data['y'], random_state=7)

"""
acc = 0
for train, test in skf.split(X, Y):
    X_train_s=X[train]
    X_test_s=X[test]
    y_train_s=Y[train]
    y_test_s=Y[test]
"""

scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train_s)
X_test_s = scaler.fit_transform(X_test_s)

# Kernel functions
y_train_s = numpy.ravel(y_train_s)
y_test_s = numpy.ravel(y_test_s)


# RBF kernel
rbf_kernel_clf = SVC(kernel='rbf')
rbf_kernel_clf.probability = True
rbf_kernel_clf.fit(X_train_s, y_train_s)
rbf_accuracy = round(rbf_kernel_clf.score(X_test_s, y_test_s),5)
#acc += rbf_accuracy
Y_predicted_s = rbf_kernel_clf.predict(X_test_s)
rbf_prf_support = precision_recall_fscore_support(y_test_s, Y_predicted_s, average = "binary")
print("Rbf kernel:")
print("Accuracy: " , rbf_accuracy, ", Precision: ", round(rbf_prf_support[0],5), ", Recall: ", round(rbf_prf_support[1],5), ", F-Score: ", round(rbf_prf_support[2],5))

cf_matrix=[]
cf_matrix = confusion_matrix(y_test_s, Y_predicted_s)

print("Confusion matrix:")
print(cf_matrix)

#print("Accuracy for SVM after cross validation: ",acc/100)

# Artificial Neural Network

X = data.drop('y',axis=1).as_matrix()
Y = data['y'].as_matrix()
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X, Y, stratify=data['y'], random_state=7)

"""
acc = 0
for train, test in skf.split(X, Y):
    X_train_a=X[train]
    X_test_a=X[test]
    y_train_a=Y[train]
    y_test_a=Y[test]
"""

#End of data preprocessing
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
scaler=scaler.fit(X_train_a)
X_train=scaler.transform(X_train_a)
scaler=scaler.fit(X_test_a)
X_test=scaler.transform(X_test_a)
###

###
mlp = nn.MLPClassifier(activation='identity',hidden_layer_sizes=(37,37,37,37,20,20,20,20,20,20,20,20,20),max_iter=1000,
    alpha=0.0001,batch_size='auto', beta_1=0.9,momentum=0.9, verbose=False,learning_rate_init=0.001)
mlp.fit(X_train_a,y_train_a)
#nn.MLPClassifier(activation='relu', alpha=0.001,batch_size='auto', beta_1=0.9,
#       beta_2=0.999, early_stopping=False, epsilon=1e-08,
#       hidden_layer_sizes=(37, 37, 37), learning_rate='adaptive',
#       learning_rate_init=0.001, max_iter=200, momentum=0.9,
#       nesterovs_momentum=True, power_t=0.5, random_state=None,
#       shuffle=True, solver='sgd', tol=0.0001, validation_fraction=0.1,
#      verbose=False, warm_start=False)


predictions = mlp.predict(X_test_a)

ann_accuracy=round(mlp.score(X_test_a, y_test_a),5)
#acc += ann_accuracy
ann_prf_support = precision_recall_fscore_support(y_test_a, predictions, average = "binary")
cf_matrix = confusion_matrix(y_test_a, predictions)

print("Artificial Neural Network:")
print("Accuracy: " , ann_accuracy, ", Precision: ", round(ann_prf_support[0],5), ", Recall: ", round(ann_prf_support[1],5), ", F-Score: ", round(ann_prf_support[2],5))
print("Confusion matrix:")
print(cf_matrix)

#print("Accuracy for ANN after cross validation: ",acc/100)

#data.hist()

fig = plt.figure(figsize=(18.0, 18.0))
ax = fig.add_subplot(111)
cax = ax.matshow(data.corr(), vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,20,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
fig.savefig('correlation.png')
#fig.show()

# Plot ROC Curve
plt.clf()
plt.figure(figsize=(8,6))

d_tree.probability = True
probas = d_tree.predict_proba(X_test_d)
fpr, tpr, thresholds = met.roc_curve(y_test_d, probas[:, 1])
roc_auc  = met.auc(fpr, tpr)
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('Decision Tree', roc_auc))

naivebayes.probability = True
probas = naivebayes.predict_proba(X_test_n)
fpr, tpr, thresholds = met.roc_curve(y_test_n, probas[:, 1])
roc_auc  = met.auc(fpr, tpr)
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('Naive Bayes', roc_auc))


probas = rbf_kernel_clf.predict_proba(X_test_s)
fpr, tpr, thresholds = met.roc_curve(y_test_s, probas[:, 1])
roc_auc  = met.auc(fpr, tpr)
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('SVM', roc_auc))

mlp.probability = True
probas = mlp.predict_proba(X_test_a)
fpr, tpr, thresholds = met.roc_curve(y_test_a, probas[:, 1])
roc_auc  = met.auc(fpr, tpr)
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('Neural Network', roc_auc))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=0, fontsize='small')
plt.show()
plt.savefig('roc.png')
