import openai
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('C:/소스 위치/embedded_lotto.csv')

df["ada_embedding"] = df.ada_embedding.apply(eval).apply(np.array)

X_train, X_test, y_train, y_test = train_test_split(list(df.ada_embedding.values), df.Num1, test_size=0.2, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(list(df.ada_embedding.values), df.Num2, test_size=0.2, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(list(df.ada_embedding.values), df.Num3, test_size=0.2, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(list(df.ada_embedding.values), df.Num4, test_size=0.2, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(list(df.ada_embedding.values), df.Num5, test_size=0.2, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(list(df.ada_embedding.values), df.Num6, test_size=0.2, random_state=42)

rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(X_train, y_train)
preds = rfr.predict(X_test)
print(preds)
print(preds[0])
print(preds.mean())

mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)

print(f"Num1 : mse={mse:.2f}, mae={mae:.2f}")
