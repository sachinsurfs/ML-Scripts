import csv
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import math

reader = csv.reader(open("spambase.data.shuffled", "rb"), delimiter=",")
data = list(reader)

training = data[:3000]
testing = data[3000:]

X = []
y = []

dim = 1

def my_custom_kernel(X,Y):
	u = 4
	norm = 2.0/(u*(u+1))
	temp= my_poly_kernel(X,Y,1) 
	x = np.empty(temp.shape)	
	for j in range(u):
		for i in range(j):
			x= np.add(my_poly_kernel(X,Y,i+j+2),x)
	return norm * x 

def my_poly_kernel(X,Y,d):
	return np.dot(X, Y.T)**d

for i in range(len(training)):
	X.append(list(training[i][:-1]))
	y.append(training[i][-1])

hx = []
y_true = []

for i in range(len(testing)):
	hx.append(list(testing[i][:-1]))
	y_true.append(testing[i][-1])


scaler = StandardScaler().fit(X)
X = scaler.transform(X)
hx = scaler.transform(hx)	


c = 64
degree = [1,2,3,4]
sv_no =[]  
msv_no = []

d_cross_error = []
d_fit_error  = []

for d in degree:
	count = 0
	clf = SVC(kernel='poly',degree=d,C=c)
	clf.fit(X, y) 
	sv_no.append(len(clf.support_vectors_))

	arr1 = clf.decision_function(clf.support_vectors_)
	for ar in arr1:
		if(abs(round(ar,5))==1.0000):
			count+=1
	msv_no.append(count)

print sv_no
print msv_no
#	cv = cross_val_score(clf, X, y,cv=10)
#	sv_no.append(clf.n_support_)
#	clf.decision_function(hx)

'''
#cv = cross_val_score(clf, X, y,cv=10)
#sv_count = clf.decision_function(hx)
y_pred = clf.predict(hx)

print accuracy_score(y_true, y_pred)

clf = SVC(kernel=my_custom_kernel,degree=d,C=c)
print clf.fit(X, y) 

y_pred = clf.predict(hx)

print accuracy_score(y_true, y_pred)
'''

#for kk in range(len(sv_count)):
#	print sv_count[kk],y_pred[kk]
#print len(clf.support_vectors_)

