# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 21:39:17 2019

@author: Bunyamin
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
#2.1. Veri Yukleme
veriler = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')

#veri on isleme
holiday=veriler.iloc[:,0:1]
temperature=veriler.iloc[:,1:2] #fahrenheit
rain_1h=veriler.iloc[:,2:3]
snow_1h=veriler.iloc[:,3:4]
clouds_all=veriler.iloc[:,4:5]
weather_main=veriler.iloc[:,5:6]
weather_description=veriler.iloc[:,6:7]
date_time=veriler.iloc[:,7:8]
traffic_volume=veriler.iloc[:,8]

#encoder:  Kategorik -> Numeric
holiday_v=holiday.values
weather_main_v=weather_main.values
weather_description_v=weather_description.values
date_time_v=date_time.values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
holiday_v[:,0] = le.fit_transform(holiday_v[:,0])
weather_main_v[:,0] = le.fit_transform(weather_main_v[:,0])
weather_description_v[:,0] = le.fit_transform(weather_description_v[:,0])
date_time_v[:,0]=le.fit_transform(date_time_v[:,0])

#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data = holiday_v, index = range(48204), columns=['tatil'] )
sonuc2 =pd.DataFrame(data =weather_main_v, index = range(48204), columns = ['hava'])
sonuc3 = pd.DataFrame(data =weather_description_v, index=range(48204), columns=['hava-aciklama'])
sonuc8= pd.DataFrame(data = date_time_v, index = range(48204), columns=['zaman'] )
#dataframe birlestirme islemi
s=pd.concat([sonuc,sonuc2],axis=1)
s2= pd.concat([s,sonuc3],axis=1)
s3=pd.concat([temperature,rain_1h],axis=1)
s4=pd.concat([s3,snow_1h],axis=1)
s5=pd.concat([s2,s4],axis=1)
s6=pd.concat([sonuc8,clouds_all],axis=1)
s7=pd.concat([s6,s5],axis=1)
#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(s5,traffic_volume,test_size=0.33, random_state=0)


#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
print("LinearReg R2 degeri:")
print(r2_score(y_test,lr.predict((X_test))))


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
x_poly = poly_reg.fit_transform(s5.values)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,traffic_volume)
print("PolynomialReg R2 degeri")
print(r2_score(traffic_volume.values,lin_reg2.predict(x_poly)))


from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=45, random_state=0)
rf_reg.fit(s5.values,traffic_volume.values)
print("RandomForest R2 degeri:")
print(r2_score(traffic_volume.values,rf_reg.predict(s5.values)))

"""
from sklearn.preprocessing import StandardScaler 

sc_X = StandardScaler() 

sc_y = StandardScaler() 

X_sc = sc_X.fit_transform(s5.values) 

y_sc = np.ravel(sc_y.fit_transform(traffic_volume.values.reshape(-1, 1)))

# Fitting SVR to the dataset 

from sklearn.svm import SVR

svr_regressor = SVR(kernel = 'rbf') 
svr_regressor.fit(X_sc,y_sc)
print("SVR")
print(r2_score(y_sc,svr_regressor.predict(X_sc)))
"""

import statsmodels.formula.api as sm 
X = np.append(arr = np.ones((48204,1)).astype(int), values=s7, axis=1 )
X_l =s7.iloc[:,[0,1,2,3,4,5,6,7]].values
r_ols = sm.OLS(endog =traffic_volume, exog =X_l.astype(float))
r = r_ols.fit()
print(r.summary())

X_l =s7.iloc[:,[1,2,3,4,5,6,7]].values
r_ols = sm.OLS(endog =traffic_volume, exog =X_l.astype(float))
r = r_ols.fit()
print(r.summary())

X_l =s7.iloc[:,[0,1,2,3,4,5,6,7]].values
r_ols = sm.OLS(endog =traffic_volume, exog =X_l.astype(float))
r = r_ols.fit()
print(r.summary())


X_l =s7.iloc[:,[1,2,3,4,5]].values
r_ols = sm.OLS(endog =traffic_volume, exog =X_l.astype(float))
r = r_ols.fit()
print(r.summary())




