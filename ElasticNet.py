import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv('csv/train.csv', index_col=False)
df_test = pd.read_csv('csv/test.csv', index_col=False)
df_train = df_train.select_dtypes(['number'])
df_test = df_test.select_dtypes(['number'])
df_test = df_test.fillna(df_test.mean())
df_train = df_train.fillna(df_train.mean())
features = list(df_train)[1:-1]
x_train = df_train[features]
y_train = df_train['SalePrice']
x_test = df_test[features]

train_scaler = StandardScaler()
train_scaler.fit(x_train)
x_train = train_scaler.transform(x_train)

test_scaler = StandardScaler()
test_scaler.fit(x_test)
x_test = test_scaler.transform(x_test)

model = ElasticNet()
y_pred = model.fit(x_train, y_train).predict(x_test)

output = pd.DataFrame({'Id': df_test['Id'], 'SalePrice': y_pred})
output.to_csv('csv/ElasticNet.csv', index=False)

# Score: 0.16431
