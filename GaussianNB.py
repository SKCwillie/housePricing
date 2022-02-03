import pandas as pd
from sklearn.naive_bayes import GaussianNB

df_train = pd.read_csv('csv/train.csv', index_col=False)
df_test = pd.read_csv('csv/test.csv', index_col=False)
df_train = df_train.select_dtypes(['number'])
df_test = df_test.select_dtypes(['number'])
features = list(df_train)[1:-1]


def normalize_test(col_name, value):
    col_max = df_test.loc[df_test[col_name].idxmax()][col_name]
    col_min = df_test.loc[df_test[col_name].idxmin()][col_name]
    return (value - col_min) / (col_max - col_min)


def normalize_train(col_name, value):
    col_max = df_test.loc[df_test[col_name].idxmax()][col_name]
    col_min = df_test.loc[df_test[col_name].idxmin()][col_name]
    return (value - col_min) / (col_max - col_min)


df_test = df_test.fillna(df_test.mean())
df_train = df_train.fillna(df_train.mean())

for feature in features:
    df_train[feature] = df_train[feature].apply(lambda x: normalize_train(feature, x))
    df_test[feature] = df_test[feature].apply(lambda x: normalize_test(feature, x))

X_train = df_train[features].to_numpy()
y_train = df_train['SalePrice'].to_numpy()
X_test = df_test[features].to_numpy()

model = GaussianNB()
y_pred = model.fit(X_train, y_train).predict(X_test)

output = pd.DataFrame({'Id': df_test['Id'], 'SalePrice': y_pred.flatten()})
output.to_csv('csv/GaussianNB.csv', index=False)
# Score: 0.25104
