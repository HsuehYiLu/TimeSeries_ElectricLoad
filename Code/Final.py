import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from scipy.stats import chi2

#read data.................................................................

df_raw_train_W = pd.read_csv('data/weatherData.csv')
df_raw_train_P = pd.read_csv('data/loadarea.csv')



df_train_P = df_raw_train_P.copy(deep=True)
df_train_W = df_raw_train_W.copy(deep=True)

df_train_W = df_train_W.rename(columns={'ObsTime': 'datetime'})
N = df_train_W[df_train_W['station'] == '466920_臺北']
C = df_train_W[df_train_W['station'] == '467490_臺中']
S = df_train_W[df_train_W['station'] == '467410_臺南']
E = df_train_W[df_train_W['station'] == '466990_花蓮']

n = df_train_P[df_train_P['area'] == 'north']
c = df_train_P[df_train_P['area'] == 'central']
s = df_train_P[df_train_P['area'] == 'south']
e = df_train_P[df_train_P['area'] == 'east']

North = pd.merge(N,n, on = 'datetime', how = 'inner')
Central = pd.merge(C,c, on = 'datetime', how = 'inner')
South = pd.merge(S,s, on = 'datetime', how = 'inner')
East = pd.merge(E,e, on = 'datetime', how = 'inner')

df = pd.concat([North, Central, South, East], sort=False)
df_save = df.copy()
#Check na values

print(df.isnull().sum())
# Drop Visb, Cloud Amount, and Sunshine, the percentage of missing value is too large
df = df.drop(columns=['Visb', 'SunShine', 'Cloud Amount'])

# check if I should replace missing value with mean or most frequent value
check_v = df['Precp'].dropna()
sns.histplot(data = check_v)
plt.title('Histogram of Precp')
plt.show()
# Most value of Precp is 0 so I decided to replace missing with zero
df['Precp'] = df['Precp'].fillna(0)
# Same way to check other features, I decided to fill na with most frequent value
df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
# Check NA
print(df.isnull().sum())

# Because the dataset included four areas data using the same time index so I decided to separate them
# Reindex dataframe with time
df_n = pd.Series(np.array(North['load']).reshape(len(North)), index=pd.date_range("1917-01-01 01:00:00", periods=len(North), freq="H"), name="load of North")
df_c = pd.Series(np.array(Central['load']).reshape(len(Central)), index=pd.date_range("1917-01-01 01:00:00", periods=len(Central), freq="H"), name="load of Central")
df_s = pd.Series(np.array(South['load']).reshape(len(South)), index=pd.date_range("1917-01-01 01:00:00", periods=len(South), freq="H"), name="load of South")
df_e = pd.Series(np.array(East['load']).reshape(len(East)), index=pd.date_range("1917-01-01 01:00:00", periods=len(East), freq="H"), name="load of East")
print(df_n.head())

# Plot dependent variable versus time
fig, ax = plt.subplots(1,1)
plt.plot(df_n, label = 'North')
plt.plot(df_c, label = 'Central')
plt.plot(df_s, label = 'South')
plt.plot(df_e, label = 'East')
plt.legend()
plt.xticks(rotation=30)
plt.xlabel('Time')
plt.ylabel('Load')
plt.title('Load vs Time')
fig.tight_layout(pad=3)
plt.show()

# ACF/PACF
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
def ACF_PACF_Plot(y,lags):
    name = y.name
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    fig.suptitle(name)
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=1)
    plt.show()

lags = 50
ACF_PACF_Plot(df_n,lags)
ACF_PACF_Plot(df_c,lags)
ACF_PACF_Plot(df_s,lags)
ACF_PACF_Plot(df_e,lags)

# From PACF, there might be a cut off at lag one or two : AR order


# Correlation Matrix with seaborn heatmap with the Pearson’s correlation coefficient.
# North
fig = plt.figure()
sns.heatmap(North.corr())
fig.tight_layout(pad=3)
fig.suptitle('Pearson’s correlation coefficient - Load of North')
plt.show()
#Central
fig = plt.figure()
sns.heatmap(Central.corr())
fig.tight_layout(pad=3)
fig.suptitle('Pearson’s correlation coefficient - Load of Central')
plt.show()
#South
fig = plt.figure()
sns.heatmap(South.corr())
fig.tight_layout(pad=3)
fig.suptitle('Pearson’s correlation coefficient - Load of South')
plt.show()
#East
fig = plt.figure()
sns.heatmap(North.corr())
fig.tight_layout(pad=3)
fig.suptitle('Pearson’s correlation coefficient - Load of East')
plt.show()

# We can see there is no big difference between each area
# Then I decided to use the North dataset as the main dataset
df_main = df.copy()
df = North.drop(columns=['Visb', 'SunShine', 'Cloud Amount'])
df['Precp'] = df['Precp'].fillna(0)
df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))


# Split the data
# from sklearn.model_selection import train_test_split  (split function from ML1)
train = df[:int(len(df)*80/100)]
test = df[int(len(df)*80/100):]

# Stationary check
# ADF/KPSS test for original dependent variable
# KPSS-test
from statsmodels.tsa.stattools import kpss
def kpss_test(timeseries):
    print('KPSS test of ' + timeseries.name)
    print('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    print(kpss_output)   #------  pvalue high station

# ADF-test
from statsmodels.tsa.stattools import adfuller
def ADF_Cal(x):
    result = adfuller(x)

    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value)) # ------- p-value low station

# difference
def difference(dataset,interval=1):
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff

# Rolling mean/variance
def Cal_rolling_mean_var(y):
    Rmean = np.zeros(len(y))
    for i in range(0,len(y)):
        Rmean[i] = y[0:i+1].mean()

    Rvar = np.zeros(len(y))
    for i in range(0,len(y)):
        Rvar[i] = np.var((y[0:i+1]))
    Rvar[0] = 0
    return Rmean, Rvar
# Main dataset
ADF_Cal(df['load'])
kpss_test(df['load'])
N_mean, N_var = Cal_rolling_mean_var(df['load'])
fig = plt.figure()
plt.subplot(211)
plt.title('Rolling mean and variance - load of North')
plt.plot(N_mean)
plt.ylabel('Rolling mean')
plt.subplot(212)
plt.plot(N_var)
plt.ylabel('Rolling var')
plt.xlabel('Index')
fig.tight_layout(pad=3)
plt.show()
# North
ADF_Cal(North['load'])
kpss_test(North['load'])
N_mean, N_var = Cal_rolling_mean_var(North['load'])
fig = plt.figure()
plt.subplot(211)
plt.title('Rolling mean and variance - load of North')
plt.plot(N_mean)
plt.ylabel('Rolling mean')
plt.subplot(212)
plt.plot(N_var)
plt.ylabel('Rolling var')
plt.xlabel('Index')
fig.tight_layout(pad=3)
plt.show()

# Central
ADF_Cal(Central['load'])
kpss_test(Central['load'])
C_mean, C_var = Cal_rolling_mean_var(Central['load'])
fig = plt.figure()
plt.subplot(211)
plt.title('Rolling mean and variance - load of Central')
plt.plot(C_mean)
plt.ylabel('Rolling mean')
plt.subplot(212)
plt.plot(C_var)
plt.ylabel('Rolling var')
plt.xlabel('Index')
fig.tight_layout(pad=3)
plt.show()

# South
ADF_Cal(South['load'])
kpss_test(South['load'])
S_mean, S_var = Cal_rolling_mean_var(South['load'])
fig = plt.figure()
plt.subplot(211)
plt.title('Rolling mean and variance - load of South')
plt.plot(S_mean)
plt.ylabel('Rolling mean')
plt.subplot(212)
plt.plot(S_var)
plt.ylabel('Rolling var')
plt.xlabel('Index')
fig.tight_layout(pad=3)
plt.show()

# Central
# South
ADF_Cal(Central['load'])
kpss_test(Central['load'])
C_mean, C_var = Cal_rolling_mean_var(Central['load'])
fig = plt.figure()
plt.subplot(211)
plt.title('Rolling mean and variance - load of Central')
plt.plot(C_mean)
plt.ylabel('Rolling mean')
plt.subplot(212)
plt.plot(C_var)
plt.ylabel('Rolling var')
plt.xlabel('Index')
fig.tight_layout(pad=3)
plt.show()

# East
ADF_Cal(East['load'])
kpss_test(East['load'])
E_mean, E_var = Cal_rolling_mean_var(East['load'])
fig = plt.figure()
plt.subplot(211)
plt.title('Rolling mean and variance - load of East')
plt.plot(E_mean)
plt.ylabel('Rolling mean')
plt.subplot(212)
plt.plot(E_var)
plt.ylabel('Rolling var')
plt.xlabel('Index')
fig.tight_layout(pad=3)
plt.show()

# Time Seires Decomposition
from statsmodels.tsa.seasonal import STL
df_n = pd.Series(np.array(df['load']).reshape(len(df)), index=pd.date_range("2017-01-01 01:00:00", periods=len(df), freq="H"), name="load of North")

STL_n = STL(df_n)
res_n = STL_n.fit()

fig = res_n.plot()
plt.show()

S = res_n.seasonal
S_adj = df_n.values - S.values

T = res_n.trend
R = res_n.resid

st = np.maximum(0,1-np.var(np.array(R))/np.var(np.array(T)+np.array(R)))
print(f'The strength of trend for this data set is {st:.2f}')

ss = np.maximum(0,1-np.var(np.array(R))/np.var(np.array(S)+np.array(R)))
print(f'The strength of seasonality for this data set is {ss:.2f}')

print('The dependent variable of area of North is highly trend and seasonal.')

# Select data by year
df_de = df_n[len(df_n) - 8760:]
STL_de = STL(df_de, period=24)
res_de = STL_de.fit()

fig = res_de.plot()
plt.show()



#Central
STL_c = STL(df_c)
res_c = STL_c.fit()

S = res_c.seasonal
S_adj = df_c.values - S.values

T = res_c.trend
R = res_c.resid

st = np.maximum(0,1-np.var(np.array(R))/np.var(np.array(T)+np.array(R)))
print(f'The strength of trend for this data set is {st:.2f}')

ss = np.maximum(0,1-np.var(np.array(R))/np.var(np.array(S)+np.array(R)))
print(f'The strength of seasonality for this data set is {ss:.2f}')

print('The dependent variable of area of Central is highly trend and seasonal.')

#South
STL_s = STL(df_s)
res_s = STL_s.fit()

S = res_s.seasonal
S_adj = df_s.values - S.values

T = res_s.trend
R = res_s.resid

st = np.maximum(0,1-np.var(np.array(R))/np.var(np.array(T)+np.array(R)))
print(f'The strength of trend for this data set is {st:.2f}')

ss = np.maximum(0,1-np.var(np.array(R))/np.var(np.array(S)+np.array(R)))
print(f'The strength of seasonality for this data set is {ss:.2f}')

print('The dependent variable of area of South is highly trend and seasonal.')

#East
STL_e = STL(df_e)
res_e = STL_e.fit()

S = res_e.seasonal
S_adj = df_e.values - S.values

T = res_e.trend
R = res_e.resid

st = np.maximum(0,1-np.var(np.array(R))/np.var(np.array(T)+np.array(R)))
print(f'The strength of trend for this data set is {st:.2f}')

ss = np.maximum(0,1-np.var(np.array(R))/np.var(np.array(S)+np.array(R)))
print(f'The strength of seasonality for this data set is {ss:.2f}')

print('The dependent variable of area of East is highly trend and seasonal.')

# We can see there is no big difference between each area
# Then I decided to use the North dataset as the main dataset
df_main = df.copy()

df = North.drop(columns=['Visb', 'SunShine', 'Cloud Amount'])
df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
df_train = df[:int(len(df)*80/100)]
df_test = df[int(len(df)*80/100):]

# Holt-Winter Method
import statsmodels.tsa.holtwinters as ets

y_HW = df_test.copy()
fitted_model = ets.ExponentialSmoothing(df_train['load'], trend = 'add', seasonal= 'add', seasonal_periods=24, damped_trend=True).fit()
y_HW['HW_fit'] = fitted_model.forecast(len(df_test))
#
#
# model = ets.ExponentialSmoothing(df_train['load'],  seasonal="add", seasonal_periods=8760)
# model2 = ets.ExponentialSmoothing(df_train['load'], trend="add", seasonal="add", seasonal_periods=12, damped_trend=True)
# fit = model.fit()
# y_HW['HW_fit'] = fit.forecast(len(df_test))
#
plt.figure()
plt.plot(df_train['load'], label='Train')
plt.plot(df_test['load'], label='Test')
plt.plot(y_HW['HW_fit'], label = 'Holt-Winter')
plt.legend(loc='best')
plt.title('Holt-Winter Seasonal forecast')
plt.xlabel('Date')
plt.ylabel('load')
plt.show()
#Last 100 data
plt.figure()
# plt.plot(df_train['load'], label='Train')
plt.plot(df_test['load'].iloc[-100:], label='Test')
plt.plot(y_HW['HW_fit'].iloc[-100:], label = 'Holt-Winter')
plt.legend(loc='best')
plt.title('Holt-Winter Seasonal forecast')
plt.xlabel('Date')
plt.ylabel('load')
plt.show()
# fig = res_n.plot()
# fig.suptitle("Trend&Seasonality&reminder")
# plt.xlabel('Time')
# plt.tight_layout()
# plt.show()


# Feature selection
from numpy import linalg as LA
df1 = df.drop(columns= ['datetime', 'station', 'area'])
X = np.matmul(df1.T, df1)
_,d,_ = np.linalg.svd(X)
print(f'singular values of Data is {d}')
print(f'The condition number of Data is {LA.cond(df1):.2f}')

# The lowest number of SVD is 2.66, although it's not close to 0 but the condition number of Data 64299 is too high.
# OLS model to decide which feature should be eliminated
y_train = df1['load']
df1 = df1.drop(columns = ['load'])
model = sm.OLS(y_train, df1).fit()
# prediction = model.predict(X_test)
print(model.summary())
print(f'AIC of model1 {model.aic:.2f}')
print(f'BIC of model1 {model.bic:.2f}')
print(f'Adjusted R-square of model1 {model.rsquared_adj:.2f}')

R1 = df1.copy()
R1 = R1.drop(columns = ['WD'])

model1 = sm.OLS(y_train, R1).fit()
model1.summary()
print(f'AIC of model1 {model1.aic:.2f}')
print(f'BIC of model1 {model1.bic:.2f}')
print(f'Adjusted R-square of model1 {model1.rsquared_adj:.2f}')

R2 = R1.copy()
R2 = R2.drop(columns = ['Td dew point'])

model2 = sm.OLS(y_train, R2).fit()
model2.summary()
print(f'AIC of model2 {model2.aic:.2f}')
print(f'BIC of model2 {model2.bic:.2f}')
print(f'Adjusted R-square of model2 {model2.rsquared_adj:.2f}')

df_drop = df.drop(columns= ['datetime', 'station', 'area', 'Td dew point', 'WD'])
X = np.matmul(df_drop.T, df_drop)
_,d,_ = np.linalg.svd(X)
print(f'singular values of Data is {d}')
print(f'The condition number of Data is {LA.cond(df_drop):.2f}')

# The condition number dropped so the direction of dropping features are correct

# try Random foreset feature selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
df1 = df.drop(columns= ['datetime', 'station', 'area'])
features = pd.get_dummies(df1)
labels = np.array(features['load'])
features= features.drop('load', axis = 1)
feature_list = list(features.columns)
features = np.array(features)
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 42)
# baseline_preds = test_features[:, feature_list.index('average')]
# baseline_errors = abs(baseline_preds - test_labels)
rf = RandomForestRegressor(n_estimators = 500, random_state = 42)
rf.fit(train_features, train_labels)

importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
# We can see that the most important feature is Temperature


df_test = test.copy()
df_train = train.copy()
# Base model
# AFM
y_avg = df_test.copy()
y_avg['avg_forcast'] = df_train['load'].mean()
sum = 0
for i in np.arange(len(test)):
    sum += (test['load'].iloc[i]- y_avg['avg_forcast'].iloc[i]) **2
    MSEtest = sum/len(test)
print(f'MSE of average forecast is {MSEtest:.2f}')


# Naive
y_nai = df_test.copy()
y_nai['NAi_forcast'] = df_train['load'].iloc[-1]

sum = 0
for i in np.arange(len(test)):
    sum += (test['load'].iloc[i]- y_nai['NAi_forcast'] .iloc[i]) **2
    MSEtest1 = sum/len(test)
print(f'MSE of Naive forecast is {MSEtest1:.2f}')


# Drift
y_D = df_test.copy()
frac = (df_train['load'].iloc[-1]) - (df_train['load'].iloc[0])
deno = 1/ (len(df_train)-1)
y_D['D_forcast'] = df_train['load'].iloc[-1]
a = []
for i in np.arange(len(y_D)):
    a.append((df_train['load'].iloc[-1]) + (frac * deno * (i+1)))
y_D['D_forcast'] = a

sum = 0
for i in np.arange(len(test)):
    sum += (test['load'].iloc[i]- y_D['D_forcast'] .iloc[i]) **2
    MSEtest2 = sum/len(test)
print(f'MSE of drift forecast is {MSEtest2:.2f}')
#SES a = 0.5
y_SES = df_train.copy()
y_SES['a_050'] = df_train['load'].iloc[0]
a_050 = [np.nan,df_train['load'].iloc[0]]
for i in np.arange(len(df_train)-2):
    a_050.append((0.5* (df_train['load'].iloc[i+1])) + (1-0.5) * (df_train['load'].iloc[i+1]))
y_SES['a_050'] = a_050
y_hat_sesf05 = df_test.copy()
y_hat_sesf05['ses_050_forcast'] = (0.5* (df_train['load'].iloc[-1])) + (1-0.5) * (df_train['load'].iloc[-1])

sum = 0
for i in np.arange(len(test)):
    sum += (test['load'].iloc[i]- y_hat_sesf05['ses_050_forcast'] .iloc[i]) **2
    MSEtest3 = sum/len(test)
print(f'MSE of SES forecast is {MSEtest3:.2f}')


# Multiple limear model
R2 = R2.iloc[:29521,:]
predict = model2.predict(R2)
plt.figure()
plt.plot(df_train['load'], label='Train')
plt.plot(predict, label='One-step')
plt.legend(loc='best')
plt.title('One-step prediction of OLS')
plt.xlabel('Date')
plt.ylabel('load')
plt.show()

df_test = test.drop(columns =  ['datetime', 'station', 'area', 'Td dew point', 'WD', 'load'])
predictions = model2.predict(df_test)
plt.figure()
plt.plot(test['load'], label='Test')
plt.plot(predictions, label='h-step')
plt.legend(loc='best')
plt.title('h-step prediction of OLS')
plt.xlabel('Date')
plt.ylabel('load')
plt.show()

sum = 0
for i in np.arange(len(test)):
    sum += (test['load'].iloc[i]- predictions.iloc[i]) **2
    MSEtest4 = sum/len(test)
print(f'MSE of OLS model forecast is {MSEtest4:.2f}')


print(f'AIC of model2 {model2.aic:.2f}')
print(f'BIC of model2 {model2.bic:.2f}')
print(f'Adjusted R-square of model2 {model2.rsquared_adj:.2f}')
print(f'Adjusted R-square of model2 {model2.rsquared:.2f}')

# Residual
res = np.zeros(len(df_train))
for i in np.arange(len(df_train)):
    res[i] = (df_train['load'][i]- predict[i])

res_mean = np.mean(res)
res_var = np.var(res)

# ACF
acf = sm.tsa.stattools.acf(res, nlags=len(res))

# Q-value (with lag 20)
Q = len(res) * np.sum(np.square(acf[20:]))


A = np.identity(len(model2.params))
A = A[1:,:]
print(model2.f_test(A))
print(model2.t_test(A))

# ARIMA, SARIMA, and ARMA
# GPAC
y_var = np.var(df_train['load'])

def get_GPAC(ry2, col, row, y_var = y_var):
    Va = ry2 * y_var
    Va = np.asarray(Va)
    mid = len(Va)//2
    place = np.array([0.0000] * (col * row))
    place = place.reshape(col, int(len(place) / col))
    place2 = place.copy()
    t = []

    for j in np.arange(row):
        for k in np.arange(1, 1 + col):
            if k == 1:
                t.append((Va[mid + j + 1]) / Va[mid + j])
            else:
                P = []
                D = []
                for i in np.arange(k):
                    P = np.append(P, Va[mid - j - i: mid - j - i + k])
                    P = P.reshape(k, int(len(P) / k))
                    D = P
                P = np.asarray(P)
                D = np.asarray(D)
                place[k - 1][j] = np.linalg.det(P)
                for e in np.arange(k):
                    D[e][-1] = Va[mid + j + e + 1]
                place2[k - 1][j] = np.linalg.det(D)
    T = place2/ place
    T[0] = t
    T = T.T
    T = T.round(3)
    x_axis_labels = np.arange(1, col + 1)
    ax = sns.heatmap(T, linewidth=0.3,xticklabels=x_axis_labels, annot=True)
    ax.set_title('Generalized Partial Autocorrelation(GPAC) Table')
    plt.show()
    print(T)

#....................................................................................

ry1 = acf[:14][::-1]
ry2 = np.concatenate((np.reshape(ry1,14), acf[:14][1:]))
get_GPAC(ry2, 7, 7)

ACF_PACF_Plot(df['load'],lags)
# Try difference 1
def difference(dataset,interval=1):
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff
d24 = difference(df['load'],interval=24)
def ACF_PACF_Plot(y,lags):
    # name = y.name
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    # fig.suptitle(name)
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=1)
    plt.show()

ACF_PACF_Plot(d24, lags)
# Try difference with 24
d24_1 = difference(d24,interval=1)
ACF_PACF_Plot(d24_1, lags)

# d1_24 seems more stationary, then try GPAC
acf1_24 = sm.tsa.stattools.acf(d24_1, nlags=len(d24_1))
ry11_24 = acf1_24[:14][::-1]
ry21_24 = np.concatenate((np.reshape(ry11_24,14), acf1_24[:14][1:]))
get_GPAC(ry21_24, 7, 7)
# From the GPAC table, I'm guessing this dataset is with no AR and MA any more
# So the original dataset might be a SARIMA(0,1,0)24


from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(df['load'], order=(24,1,0))
model_fit = model.fit()
print(model_fit.summary())
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
residuals.plot(kind='kde')
plt.title('Density of residual')
plt.show()



model_hat = model_fit.predict(start = 0, end = len(df)-1)
VAR = np.var(model_hat)
errortrain1 = [np.nan]
for i in np.arange(len(train)-1):
        errortrain1.append(df_train['load'].iloc[i+1]- model_hat.iloc[i+1])
print(f'Variance of residual is {np.var(errortrain1[1:]):.2f}')

test_error = test.copy()
sum = 0
for i in np.arange(len(test)):
    sum += (test_error['load'].iloc[i]- model_hat.iloc[i + len(df_train)]) **2
    MSEtest6 = sum/len(test)
error = []
for i in np.arange(len(test)):
    error.append(test['load'].iloc[i]- model_hat.iloc[i + len(df_train)])
print(f'Variance of forecast error of ARIMA is {np.var(error):.2f}')

plt.figure()
plt.plot(df_train['load'], label='Train')
plt.plot(test['load'], label='Test')
plt.plot(model_hat, label='prediction of ARIMA')
plt.legend(loc='best')
plt.title('One-step ahead prediction')
plt.xlabel('t')
plt.ylabel('yt')
plt.show()

# First 100 prediction
plt.figure()
plt.plot(df_train['load'][:100], label='Train')
# plt.plot(y_test, label='Test')
plt.plot(model_hat[:100], label='one-step ahead prediction')
plt.legend(loc='best')
plt.title('One-step ahead prediction')
plt.xlabel('t')
plt.ylabel('yt')
plt.show()
# Most prediction fit the dataset
from scipy.stats import chi2
N = len(df)
lb_stat, pvalue = sm.stats.acorr_ljungbox(errortrain1[1:], lags=[20], return_df=True).iloc[0]
na = 24
nb = 0
alfa = 0.01
DOF = N - na - nb
chi_critical = chi2.ppf(1 - alfa, DOF)
if lb_stat < chi_critical:
    print('The residual is white.')
if lb_stat > chi_critical:
    print('The residual is not white.')


# lm method
from scipy import signal
na = 1
nb = 0
y = df_train['load']
def cal_e(seta,y = y,na = na, nb = nb):
    max_order = max(na, nb)
    num = [0.00] * (max_order)
    den = [0.00] * (max_order)
    num = np.asarray(num)
    den = np.asarray(den)
    for i in np.arange(len(seta)-nb):
        if len(seta)-nb ==0:
            den = den
        else:
            den[i] = seta[i]
    for i in np.arange(len(seta)-na):
        if len(seta)-na == 0:
            num = num
        else:
            num[i] = seta[na+i]
    ar = np.r_[1, den]
    ma = np.r_[1, num]
    sys = (ar,ma,1)
    _,e = signal.dlsim(sys,y)
    return e

def x_dev(seta, tel =10 ** (-5), y =y, na =na, nb = nb, cal_e = cal_e):
    X = []
    e = cal_e(seta,y = y,na = na, nb = nb)
    for i in np.arange(na+nb):
        seta_add = seta.copy()
        seta_add[i] = seta[i] + tel
        d = cal_e(seta_add, y, na, nb) - e
        X = np.append(X, (d/tel).T)
    # X = X.reshape(len(y), na+nb)
    X = X.reshape(na + nb, len(y))
    # A = np.dot(X.T, X)
    A = np.dot(X, X.T)
    # g = np.dot(X.T, e)
    g = np.dot(X, e)
    return A,g


def LM(y, na,nb, max = 100, mu = 10 ** (-3), x_dev= x_dev,  cal_e = cal_e):
    k = 0
    seta = np.zeros(na+nb)
    mu = mu
    I = np.identity(na + nb)
    SSE_record = []
    while k<max:
        k+= 1
        e = cal_e(seta, y=y, na=na, nb = nb)
        SSE = np.dot(e.T,e)
        SSE = np.asscalar(SSE)
        A,g = x_dev(seta, tel=10 ** (-6), y=y, na=na, nb=nb)
        del_seg = np.dot(np.linalg.inv((A + mu*I)),g)
        del_seg = del_seg.T
        new_seta = seta - del_seg
        new_seta = new_seta.reshape(na+nb,)
        new_e = cal_e(new_seta, y=y, na=na, nb =nb)
        SSE_new = np.dot(new_e.T,new_e)
        SSE_new = np.asscalar(SSE_new)
        SSE_record.append(SSE)
        if SSE_new < SSE:
            if np.linalg.norm(del_seg) < 10 ** (-3):
                seta_end = new_seta
                z = cal_e(seta_end, y=y, na=na)
                ez = np.dot(z.T,z)
                ez = np.asscalar(ez)
                cov = ez/ (len(y)-(na+nb))
                Cov = np.dot(cov, np.linalg.inv(A))
                reason = ' delta theta < 10 ** (-3)'
                return seta_end, reason, Cov, new_e, SSE_record
            else :
                seta = new_seta
                mu = mu /10
        elif SSE_new > SSE:
            mu = mu*10
            if mu > 10**9:
                reason = 'Mu is too large'
                return seta, reason
    if k >= max:
        reason = 'k larger than max'
        return seta, reason


# teta,_,Cov, new_e, SSE_record = LM(df_train['load'], na,nb, max = 100, mu = 10 ** (-3), x_dev= x_dev,  cal_e = cal_e)


Cov = model_fit.cov_params()


# LSTM

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

scaler = MinMaxScaler(feature_range=(0, 1))
df_LS =df['load']
dataset = df_LS.values
dataset = dataset.astype('float32')
dataset = scaler.fit_transform(dataset.reshape(-1,1))


train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=2)


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset), label = 'Data')
plt.plot(trainPredictPlot, label = 'trainPredict')
plt.plot(testPredictPlot, label = 'testPredict')
plt.title('LSTM prediction')
plt.legend()
plt.show()

error = []
for i in np.arange(len(testPredict)):
    error.append(test['load'].iloc[i]- testPredict[i])
print(f'Variance of prediction error of LSTM is {np.var(error):.2f}')


# Forecast Function
# model_fit.seasonalarparams

# h-step prediction of SARIMA

plt.figure()
plt.plot(test['load'], label='Test')
plt.plot(model_hat[len(df_train):], label='prediction of SARIMA')
plt.legend(loc='best')
plt.title('h-step ahead prediction')
plt.xlabel('t')
plt.ylabel('load')
plt.show()