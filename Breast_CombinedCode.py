import sklearn
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#Load data
dataset_Mal_Mohammadi = np.loadtxt(open('/My Drive/Breast 1/Data/Robust_Radiomics_Malaysia_Mohammadi.csv', "rb"), delimiter=",")

dataset_Irn_Mohammadi = np.loadtxt(open('/My Drive/Breast 1/Data/Robust_Radiomics_Iran_Mohammadi.csv', "rb"), delimiter=",")

dataset_Turkish = np.loadtxt(open('/My Drive/Breast 1/Data/Robust_Radiomics_Turkish_Taha.csv', "rb"), delimiter=",")

dataset_Mal_Mohammadi = np.random.permutation(dataset_Mal_Mohammadi)
dataset_Mal_Mohammadi = np.random.permutation(dataset_Mal_Mohammadi)

dataset_Irn_Mohammadi = np.random.permutation(dataset_Irn_Mohammadi)
dataset_Irn_Mohammadi = np.random.permutation(dataset_Irn_Mohammadi)

dataset_Turkish = np.random.permutation(dataset_Turkish)
dataset_Turkish = np.random.permutation(dataset_Turkish)

X_Mal_Mohammadi = dataset_Mal_Mohammadi[:,1:]
y_Mal_Mohammadi = dataset_Mal_Mohammadi[:,0]

X_Ir_Mohammadi = dataset_Irn_Mohammadi[:,1:]
y_Ir_Mohammadi = dataset_Irn_Mohammadi[:,0]

X_Turkish = dataset_Turkish[:,1:]
y_Turkish = dataset_Turkish[:,0]

#np.save('/My Drive/Breast 1/Data/X_Ir_Mohammadi.npy',X_Ir_Mohammadi)
#np.save('/My Drive/Breast 1/Data/y_Ir_Mohammadi.npy',y_Ir_Mohammadi)

#np.save('/My Drive/Breast 1/Data/X_Mal_Mohammadi.npy',X_Mal_Mohammadi)
#np.save('/My Drive/Breast 1/Data/y_Mal_Mohammadi.npy',y_Mal_Mohammadi)

#np.save('/My Drive/Breast 1/Data/X_Turkish.npy',X_Turkish)
#np.save('/My Drive/Breast 1/Data/y_Turkish.npy',y_Turkish)

X_Mal_Mohammadi = np.load('/gdrive/My Drive/Breast 1/Data/X_Mal_Mohammadi.npy')
y_Mal_Mohammadi = np.load('/gdrive/My Drive/Breast 1/Data/y_Mal_Mohammadi.npy')

X_Turkish = np.load('/gdrive/My Drive/Breast 1/Data/X_Turkish.npy')
y_Turkish = np.load('/gdrive/My Drive/Breast 1/Data/y_Turkish.npy')

X_Ir_Mohammadi = np.load('/gdrive/My Drive/Breast 1/Data/X_Ir_Mohammadi.npy')
y_Ir_Mohammadi = np.load('/gdrive/My Drive/Breast 1/Data/y_Ir_Mohammadi.npy')

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

min_max_scaler.fit(X_Mal_Mohammadi)
X_Mal_Mohammadi = min_max_scaler.transform(X_Mal_Mohammadi)

min_max_scaler.fit(X_Ir_Mohammadi)
X_Ir_Mohammadi = min_max_scaler.transform(X_Ir_Mohammadi)

min_max_scaler.fit(X_Turkish)
X_Turkish = min_max_scaler.transform(X_Turkish)

X_Train, X_Test_Malaysia, y_Train, y_Test_Malaysia = train_test_split(X_Mal_Mohammadi, y_Mal_Mohammadi, test_size=0.85, shuffle=True, stratify=y_Mal_Mohammadi)
X_Train, X_Val_Malaysia, y_Train, y_Val_Malaysia = train_test_split(X_Train, y_Train, test_size=0.15, shuffle=True, stratify=y_Train)

X_Train_Testing = X_Train
y_Train_Testing = y_Train

X_Test_Turkish = X_Turkish
y_Test_Turkish = y_Turkish

X_Test_Iran = X_Ir_Mohammadi
y_Test_Iran = y_Ir_Mohammadi

np.savetxt("/My Drive/Breast 1/ResultData/y_Train_Testing.csv", y_Train_Testing, delimiter=",")
np.savetxt("/My Drive/Breast 1/ResultData/y_Test_Malaysia.csv", y_Test_Malaysia, delimiter=",")
np.savetxt("/My Drive/Breast 1/ResultData/y_Test_Iran.csv", y_Test_Iran, delimiter=",")
np.savetxt("/My Drive/Breast 1/ResultData/y_Test_Turkish.csv", y_Test_Turkish, delimiter=",")

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
param_grid = {"max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              "min_samples_split": [5, 10],
              "n_estimators": [5, 10, 20, 30, 40, 50, 70, 80, 100, 130, 150],
              "criterion": ['friedman_mse', 'squared_error', 'mse'],
              "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
              "max_features": ['auto','sqrt','log2']}

search = GridSearchCV(gbc, param_grid).fit(X_Val_Malaysia, y_Val_Malaysia)
print(search.best_params_)

gbc = GradientBoostingClassifier(loss= 'deviance',learning_rate=0.1, n_estimators=50, criterion= 'friedman_mse')
gbc.fit(X_Train, y_Train)

print("Malaysian Training:")
y_pred = gbc.predict(X_Train_Testing)
y_predProbability = gbc.predict_proba(X_Train_Testing)
np.savetxt("/My Drive/Breast 1/ResultData/y_predProbability_XGBoost_TrainTesting.csv", y_predProbability, delimiter=",")

y_orig = y_Train_Testing

tn, fp, fn, tp = confusion_matrix(y_orig, y_pred).ravel()
print("TP: ", tp)
print("TN: ", tn)
print("FP: ", fp)
print("FN: ", fn)

specificity = tn / (tn+fp)
Sensitivity = tp / (tp + fn)
print("Sensitivity: ", round(Sensitivity,3))
print("Specificity: ", round(specificity,3))


print("AUC: ", round(roc_auc_score(y_orig, y_pred),3))


print("Malaysian Testing:")
y_pred = gbc.predict(X_Test_Malaysia)
y_predProbability = gbc.predict_proba(X_Test_Malaysia)

np.savetxt("/My Drive/Breast 1/ResultData/y_predProbability_XGBoost_MalaysianTesting.csv", y_predProbability, delimiter=",")

y_orig = y_Test_Malaysia

tn, fp, fn, tp = confusion_matrix(y_orig, y_pred).ravel()
print("TP: ", tp)
print("TN: ", tn)
print("FP: ", fp)
print("FN: ", fn)

specificity = tn / (tn+fp)
Sensitivity = tp / (tp + fn)
print("Sensitivity: ", round(Sensitivity,3))
print("Specificity: ", round(specificity,3))

print("AUC: ", round(roc_auc_score(y_orig, y_pred),3))

print("IRAN Testing:")
y_pred = gbc.predict(X_Test_Iran)
y_predProbability = gbc.predict_proba(X_Test_Iran)
np.savetxt("/My Drive/Breast 1/ResultData/y_predProbability_XGBoost_IranTesting.csv", y_predProbability, delimiter=",")

y_orig = y_Test_Iran

tn, fp, fn, tp = confusion_matrix(y_orig, y_pred).ravel()
print("TP: ", tp)
print("TN: ", tn)
print("FP: ", fp)
print("FN: ", fn)

specificity = tn / (tn+fp)
Sensitivity = tp / (tp + fn)
print("Sensitivity: ", round(Sensitivity,3))
print("Specificity: ", round(specificity,3))

print("AUC: ", round(roc_auc_score(y_orig, y_pred),3))

print("Turkish testing:")
y_pred = gbc.predict(X_Test_Turkish)
y_predProbability = gbc.predict_proba(X_Test_Turkish)

np.savetxt("/My Drive/Breast 1/ResultData/y_predProbability_XGBoost_TurkishTesting.csv", y_predProbability, delimiter=",")

y_orig = y_Test_Turkish

tn, fp, fn, tp = confusion_matrix(y_orig, y_pred).ravel()
print("TP: ", tp)
print("TN: ", tn)
print("FP: ", fp)
print("FN: ", fn)
specificity = tn / (tn + fp)
Sensitivity = tp / (tp + fn)
print("Sensitivity: ", round(Sensitivity,3))
print("Specificity: ", round(specificity,3))

print("AUC: ", round(roc_auc_score(y_orig, y_pred),3))


#Random Forest Classifier Train Test Splitting
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
param_grid = {"max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              "min_samples_split": [2, 3, 4, 5],
              "n_estimators": [5, 10, 15, 20, 25, 30, 35, 40, 50, 70, 80, 100, 130, 150],
              "criterion": ['gini', 'entropy', 'log_loss'],
              "max_features": ['sqrt','log2', 'None']}

search = GridSearchCV(rfc, param_grid).fit(X_Val_Malaysia, y_Val_Malaysia)
print(search.best_params_)

rfc = RandomForestClassifier(n_estimators=15, criterion='gini', max_features= 'None', max_depth = 5, min_samples_split=2)

rfc.fit(X_Train, y_Train)

print("Malaysian Training:")
y_pred = rfc.predict(X_Train_Testing)
y_predProbability = rfc.predict_proba(X_Train_Testing)

np.savetxt("/My Drive/Breast 1/ResultData/y_predProbability_RandomForest_TrainTesting.csv", y_predProbability, delimiter=",")

y_orig = y_Train_Testing

tn, fp, fn, tp = confusion_matrix(y_orig, y_pred).ravel()
print("TP: ", tp)
print("TN: ", tn)
print("FP: ", fp)
print("FN: ", fn)

specificity = tn / (tn+fp)
Sensitivity = tp / (tp + fn)
print("Sensitivity: ", round(Sensitivity,3))
print("Specificity: ", round(specificity,3))

print("AUC: ", round(roc_auc_score(y_orig, y_pred),3))


print("Malaysian Testing:")
y_pred = rfc.predict(X_Test_Malaysia)
y_predProbability = rfc.predict_proba(X_Test_Malaysia)

np.savetxt("/My Drive/Breast 1/ResultData/y_predProbability_RandomForest_MalaysianTesting.csv", y_predProbability, delimiter=",")

y_orig = y_Test_Malaysia

tn, fp, fn, tp = confusion_matrix(y_orig, y_pred).ravel()
print("TP: ", tp)
print("TN: ", tn)
print("FP: ", fp)
print("FN: ", fn)
specificity = tn / (tn+fp)
Sensitivity = tp / (tp + fn)
print("Sensitivity: ", round(Sensitivity,3))
print("Specificity: ", round(specificity,3))

print("AUC: ", round(roc_auc_score(y_orig, y_pred),3))


print("IRN testing:")
y_pred = rfc.predict(X_Test_Iran)
y_predProbability = rfc.predict_proba(X_Test_Iran)

np.savetxt("/My Drive/Breast 1/ResultData/y_predProbability_RandomForest_IranTesting.csv", y_predProbability, delimiter=",")

y_orig = y_Test_Iran

tn, fp, fn, tp = confusion_matrix(y_orig, y_pred).ravel()
print("TP: ", tp)
print("TN: ", tn)
print("FP: ", fp)
print("FN: ", fn)

specificity = tn / (tn+fp)
Sensitivity = tp / (tp + fn)
print("Sensitivity: ", round(Sensitivity,3))
print("Specificity: ", round(specificity,3))

print("AUC: ", round(roc_auc_score(y_orig, y_pred),3))

print("Turkish testing:")
y_pred = rfc.predict(X_Test_Turkish)
y_predProbability = rfc.predict_proba(X_Test_Turkish)
np.savetxt("/gdrive/My Drive/Breast 1/ResultData/y_predProbability_RandomForest_TurkishTesting.csv", y_predProbability, delimiter=",")

y_orig = y_Test_Turkish

tn, fp, fn, tp = confusion_matrix(y_orig, y_pred).ravel()
print("TP: ", tp)
print("TN: ", tn)
print("FP: ", fp)
print("FN: ", fn)
specificity = tn / (tn+fp)
Sensitivity = tp / (tp + fn)
print("Sensitivity: ", round(Sensitivity,3))
print("Specificity: ", round(specificity,3))

print("AUC: ", round(roc_auc_score(y_orig, y_pred),3))

#SVC 
from sklearn.svm import SVC
svc = SVC()
param_grid = {"C": [1, 2, 3],
              "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
              "degree": [2,3],
              "gamma": ['scale', 'auto']}

search = GridSearchCV(svc, param_grid).fit(X_Val_Malaysia, y_Val_Malaysia)
print(search.best_params_)

svc = SVC(kernel='poly', degree=2, C=2, gamma='auto', probability=True)

svc.fit(X_Train, y_Train)

print("Malaysian Training:")
y_pred = svc.predict(X_Train_Testing)
y_predProbability = svc.predict_proba(X_Train_Testing)
np.savetxt("/My Drive/Breast 1/ResultData/y_predProbability_SVM_TrainTesting.csv", y_predProbability, delimiter=",")

y_orig = y_Train_Testing

tn, fp, fn, tp = confusion_matrix(y_orig, y_pred).ravel()
print("TP: ", tp)
print("TN: ", tn)
print("FP: ", fp)
print("FN: ", fn)

specificity = tn / (tn+fp)
Sensitivity = tp / (tp + fn)
print("Sensitivity: ", round(Sensitivity,3))
print("Specificity: ", round(specificity,3))

print("AUC: ", round(roc_auc_score(y_orig, y_pred),3))

print("Malaysian Testing:")

y_pred = svc.predict(X_Test_Malaysia)
y_predProbability = svc.predict_proba(X_Test_Malaysia)
np.savetxt("/My Drive/Breast 1/ResultData/y_predProbability_SVM_MalaysianTesting.csv", y_predProbability, delimiter=",")


y_orig = y_Test_Malaysia

tn, fp, fn, tp = confusion_matrix(y_orig, y_pred).ravel()
print("TP: ", tp)
print("TN: ", tn)
print("FP: ", fp)
print("FN: ", fn)

specificity = tn / (tn+fp)
Sensitivity = tp / (tp + fn)
print("Sensitivity: ", round(Sensitivity,3))
print("Specificity: ", round(specificity,3))

from sklearn.metrics import roc_auc_score
print("AUC: ", round(roc_auc_score(y_orig, y_pred),3))

print("IRN testing:")
y_pred = svc.predict(X_Test_Iran)
y_predProbability = svc.predict_proba(X_Test_Iran)
np.savetxt("/My Drive/Breast 1/ResultData/y_predProbability_SVM_IranTesting.csv", y_predProbability, delimiter=",")

y_orig = y_Test_Iran

tn, fp, fn, tp = confusion_matrix(y_orig, y_pred).ravel()
print("TP: ", tp)
print("TN: ", tn)
print("FP: ", fp)
print("FN: ", fn)

specificity = tn / (tn+fp)
Sensitivity = tp / (tp + fn)
print("Sensitivity: ", round(Sensitivity,3))
print("Specificity: ", round(specificity,3))

print("AUC: ", round(roc_auc_score(y_orig, y_pred),3))

print("Turkish testing:")
y_pred = svc.predict(X_Test_Turkish)
y_predProbability = svc.predict_proba(X_Test_Turkish)
np.savetxt("/My Drive/Breast 1/ResultData/y_predProbability_SVM_TurkishTesting.csv", y_predProbability, delimiter=",")

y_orig = y_Test_Turkish

tn, fp, fn, tp = confusion_matrix(y_orig, y_pred).ravel()
print("TP: ", tp)
print("TN: ", tn)
print("FP: ", fp)
print("FN: ", fn)
specificity = tn / (tn+fp)
Sensitivity = tp / (tp + fn)
print("Sensitivity: ", round(Sensitivity,3))
print("Specificity: ", round(specificity,3))

from sklearn.metrics import roc_auc_score
print("AUC: ", round(roc_auc_score(y_orig, y_pred),3))

#######################################################################################

