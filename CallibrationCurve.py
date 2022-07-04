import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.metrics import roc_curve
from sklearn.calibration import calibration_curve

df=pd.read_csv('AllCohorts.csv', sep=',',header=None)
data = df.values
print(data)

y_true = data[:,3]
y_pred1 = data[:,0]
y_pred2 = data[:,1]
y_pred3 = data[:,2]

########BS
N = y_true.shape[0]
S1 = 0
S2 = 0
S3 = 0
for i in range(N):
	S1 = S1 + ((y_pred1[i] - y_true[i])**2)
	S2 = S2 + ((y_pred2[i] - y_true[i])**2)
	S3 = S3 + ((y_pred3[i] - y_true[i])**2)
	
BS1 = round(S1/N,3)
BS2 = round(S2/N,3)
BS3 = round(S3/N,3)

print(BS1)
print(BS2)
print(BS3)
########End of BS

prob_true1, prob_pred1 = calibration_curve(y_true, y_pred1, n_bins=5)
prob_true2, prob_pred2 = calibration_curve(y_true, y_pred2, n_bins=5)
prob_true3, prob_pred3 = calibration_curve(y_true, y_pred3, n_bins=5)

print(prob_true1)
print(prob_pred1)

ns_probs = [0 for _ in range(5)]

trueLabels = np.zeros(5,int)
trueLabels[0:3] = 1

ns_fpr, ns_tpr, _ = roc_curve(trueLabels, ns_probs)

pyplot.plot(ns_fpr, ns_tpr, linestyle='--')
pyplot.plot(prob_pred1, prob_true1, color='blue', marker='*', label='XGBoost (Brier Score=0.145)')
pyplot.plot(prob_pred2, prob_true2, color='green', marker='s', label='Random Forest (Brier Score=0.157)')
pyplot.plot(prob_pred3, prob_true3, color='orange', marker='o', label='SVM (Brier Score=0.222)')
#pyplot.plot(lr_fpr2, lr_tpr2, marker='.', label='Logistic')

pyplot.title('All Cohorts')

# axis labels
pyplot.xlabel('Predicted Probability')
pyplot.ylabel('Observed Frequency')
# show the legend
pyplot.gca().set_aspect('equal', adjustable='box')
pyplot.legend()
pyplot.legend(fontsize=8)
# show the plot
pyplot.savefig('AllCohorts.png',dpi=600)

pyplot.show()


