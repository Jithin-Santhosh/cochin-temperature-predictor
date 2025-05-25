import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

#Loading Data
df=pd.read_csv("ap.csv",parse_dates=['DATE'])
df['Day']=df['DATE'].dt.day
df['Month']=df['DATE'].dt.month
df['Year']=df['DATE'].dt.year

x=df[['Day','Month','Year']]
y=df['Max. Temperature']

model=LinearRegression()
model.fit(x,y)
joblib.dump(model,'temp_model.pkl')
