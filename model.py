import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import pickle
# Load the dataset
df = pd.read_csv('kc_house_data.csv')
print(df.head())
# Data preprocessing
df=df.drop(columns=['id','date'])
log_price=np.log1p(df['price'])
x=df.drop(columns=['price'])
y=log_price
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
# Train the model
model=LinearRegression()
model.fit(x_train,y_train)  
# Predict and evaluate the model
y_pred=model.predict(x_test)    
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
print(f'Mean Absolute Error: {mae}')
# Cross-validation
cv_scores=cross_val_score(model,x,y,cv=5,scoring='r2')
print(f'Cross-validation scores: {cv_scores}')
print(f'Average Cross-validation score: {np.mean(cv_scores)}')
pred_price = np.expm1(y_pred)
actual_price = np.expm1(y_test)

mae_original = mean_absolute_error(
    actual_price,
    pred_price
)

rmse_original = np.sqrt(
    mean_squared_error(
        actual_price,
        pred_price
    )
)

print(f"Original Scale MAE: {mae_original}")
print(f"Original Scale RMSE: {rmse_original}")


with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

