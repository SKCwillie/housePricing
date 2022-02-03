import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.svm import SVR, LinearSVR
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# Import Data, Select Number Columns, and split into X/y
df = pd.read_csv('csv/train.csv', index_col=False)
df_test = pd.read_csv('csv/test.csv', index_col=False)
df = df.select_dtypes(['number'])
df = df.fillna(df.mean())
features = list(df)[1:-1]
X = df[features]
y = df['SalePrice']

# Scale and Split Data to train/test
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)

# Create and Score Linear Models
linear_models = {'ElasticNet': ElasticNet(), 'Lasso': Lasso(), 'GaussianNB': GaussianNB(),
                 'LinearRegression': LinearRegression(), 'Ridge': Ridge(), 'SVR': SVR(kernel='linear'),
                 'LinearSVR': LinearSVR()}

linear_scores = []
max_linear = 0

for i in linear_models:
    model = linear_models[i]
    y_pred = model.fit(X_train, y_train).predict(X_test)
    score = model.score(X_train, y_train)
    print(f' {i} score: {score}')
    linear_scores.append(score)
    if score > max_linear:
        max_linear = score
        max_linear_model = i


