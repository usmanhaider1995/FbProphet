# FbProphet Library
# conda install -c conda-forge fbprophet
import fbprophet
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric


df = pd.read_csv('airline_passengers.csv')
#print(df.head())

#print(df.plot())
df.columns =['ds','y']

df.drop(144,axis=0,inplace=True)
#print(df.tail())
df['Date'] = pd.to_datetime(df['ds'])
df.drop('Date',axis=1 ,inplace=True)
#print(df.tail())
#print(dir(Prophet))
model =Prophet()

model.fit(df)
#print(model.component_modes)
# make extra 365 days and append it to previous dataset

future_date = model.make_future_dataframe(periods=365)
print(future_date.tail())
prediction = model.predict(future_date)
print(prediction.head())
print(prediction.columns)
model.plot(prediction)

#Visualization for each component

model.plot_components(prediction)
# cross validation take first data as training second data as testing and last one for validation (prediction)

df_cv = cross_validation(model,initial='730 days', period='180 days', horizon='365 days')
print(df_cv.head())

df_p = performance_metrics(df_cv)
print(df_p.head())
fig = plot_cross_validation_metric(df_cv, metric='rmse')
































plt.show()


