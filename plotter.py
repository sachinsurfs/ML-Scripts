import csv
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import math



cross_error = [0.072999999999999843, 0.091999999999999971, 0.098000000000000087, 0.17099999999999993]
fit_error = [0.068707058088694595, 0.086196127420362312, 0.091193004372267294, 0.15802623360399748]
vc = [618, 802, 982, 1344]
degree = [1,2,3,4]


sd = [0.025044405008349098, 0.01292714628644352, 0.015205992970609381, 0.013601470508735431, 0.01631631766736057, 0.013482498944104459, 0.014757295747452446, 0.014487542541400505, 0.013097921802925664, 0.012827920936595913, 0.013370780746754388, 0.012206555615733696, 0.012128936932440163, 0.011780398031381527, 0.013349989596333841]
AccList = [0.26499999999999996, 0.17866666666666661, 0.14233333333333337, 0.12166666666666667, 0.10266666666666668, 0.092666666666666661, 0.086666666666666684, 0.076333333333333336, 0.074666666666666687, 0.075666666666666688, 0.076333333333333336, 0.075666666666666674, 0.074666666666666645, 0.072999999999999995, 0.074666666666666687]



#sd = [0.0042687494916219173, 0.0083266639978645113, 0.0073105707331537605, 0.013012814197295428, 0.01744833643773654, 0.017178798302300163, 0.016478942792411015, 0.012274635093014642, 0.013498971154211057, 0.0071180521680208704, 0.011855612829185824, 0.015438048235879227, 0.014910846164230053, 0.016478942792411036, 0.0089690826980491356]
#AccList = [0.38133333333333336, 0.3613333333333334, 0.33900000000000008, 0.3086666666666667, 0.27666666666666667, 0.23933333333333334, 0.19800000000000001, 0.16933333333333334, 0.13999999999999999, 0.11599999999999996, 0.098333333333333328, 0.088333333333333319, 0.090999999999999998, 0.091999999999999998, 0.09133333333333335]


#sd =[0.005783117190965836, 0.0090921211313238787, 0.007717224601860165, 0.012775845264491199, 0.016055459438389711, 0.015491933384829676, 0.018147543451754917, 0.015691469727919745, 0.018574175621006706, 0.017130220210039467, 0.013597385369580757, 0.009433981132056608, 0.0067986926847903697, 0.0076303487615063765, 0.0093630479367920981]
#AccList = [0.37566666666666659, 0.36133333333333328, 0.34933333333333327, 0.33033333333333326, 0.30666666666666664, 0.27999999999999997, 0.26533333333333331, 0.23600000000000004, 0.20833333333333331, 0.17900000000000005, 0.15133333333333332, 0.13100000000000001, 0.10933333333333332, 0.098000000000000004, 0.0896666666666667]

#sd = [0.0064031242374328482, 0.0087496031656044641, 0.0081921371516296485, 0.0088191710368819756, 0.016155494421403519, 0.015452435981999017, 0.015040685563571318, 0.0141578403877302, 0.012476644848141934, 0.01699999999999998, 0.01468181036369684, 0.015351438586225936, 0.016519348924485137, 0.012476644848141925, 0.012845232578665138]
#AccList = [0.373, 0.36633333333333334, 0.35533333333333333, 0.34000000000000002, 0.33033333333333337, 0.31033333333333335, 0.29599999999999999, 0.27866666666666667, 0.26100000000000001, 0.25099999999999995, 0.22333333333333333, 0.20566666666666672, 0.184, 0.17100000000000001, 0.15166666666666667]


sd = [x * 100 for x in sd]
AccList = [x * 100 for x in AccList]


CList = [0.0078125,0.015625,0.03125,0.0625,0.125,0.25,0.5,1,2,4,8,16,32,64,128]


x = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
ap = [0.3,0.46,0.657,0.96,0.98,1,1,1,1,1,1]
ada = [0.17,0.28,0.42,0.56,0.8,0.97,1,1,1,1,1]

#ada = [0.26,0.34,0.38,0.40,0.47,0.6,0.065,0.78,0.85,1]

plt.plot(x, ap)
plt.plot(x, ada)

#plt.errorbar(CList,AccList,yerr=sd)

plt.title("Cumulative Margin")
plt.ylabel("% below theta")
plt.xlabel("theta")
plt.show()




