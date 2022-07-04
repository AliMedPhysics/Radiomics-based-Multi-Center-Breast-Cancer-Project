import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix

def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
	net_benefit_model = np.array([])
	for thresh in thresh_group:
		y_pred_label = y_pred_score > thresh
		tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
		n = len(y_label)
		net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
		net_benefit_model = np.append(net_benefit_model, net_benefit)
	return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_label):
	net_benefit_all = np.array([])
	tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
	total = tp + tn
	for thresh in thresh_group:
		net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
		net_benefit_all = np.append(net_benefit_all, net_benefit)
	return net_benefit_all


#Start of Calculation
df=pd.read_csv('AllCohorts.csv', sep=',',header=None)
data = df.values


y_label = data[:,3]
y_pred_score1 = data[:,0]
y_pred_score2 = data[:,1]
y_pred_score3 = data[:,2]

thresh_group = np.arange(0,1,0.01)

net_benefit_model1 = calculate_net_benefit_model(thresh_group, y_pred_score1, y_label)
net_benefit_model2 = calculate_net_benefit_model(thresh_group, y_pred_score2, y_label)
net_benefit_model3 = calculate_net_benefit_model(thresh_group, y_pred_score3, y_label)
net_benefit_all = calculate_net_benefit_all(thresh_group, y_label)


pyplot.plot(thresh_group, net_benefit_all, color = 'black',label = 'Intervention for all')
pyplot.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Intervention for none')
pyplot.plot(thresh_group, net_benefit_model1, color='blue', label='XGBoost')
pyplot.plot(thresh_group, net_benefit_model2, color='green', label='Random Forest')
pyplot.plot(thresh_group, net_benefit_model3, color='orange', label='SVM')

pyplot.xlim(0,1)
pyplot.ylim(net_benefit_model1.min() - 0.15, net_benefit_model1.max() + 0.15)

pyplot.title('All Cohorts')

# axis labels
pyplot.xlabel('Threshold Probability')
pyplot.ylabel('Net Benefit')
# show the legend
pyplot.gca().set_aspect('equal', adjustable='box')
pyplot.legend()
pyplot.legend(fontsize=10)
# show the plot
pyplot.savefig('AllCohorts.png',dpi=600)

pyplot.show()
