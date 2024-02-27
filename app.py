# Delete Columns
df.drop(columns=['Adj Close', 'Volume'], inplace=True)


# MinMax Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['Open', 'High', 'Low', 'Close']] = pd.DataFrame(scaler.fit_transform(df[['Open', 'High', 'Low', 'Close']]), columns=df[['Open', 'High', 'Low', 'Close']].columns)


# Feature Extraction - Addition
df[close_predict] = df[Close] + df[Select]


# Feature Extraction - Addition
df[close_predict] = df[Close] + df[Open]


# Data Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Close', axis=1), df['Close'], train_size=0.7, random_state=42)


# Model Building --> Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True, positive=True, copy_X=True)
model.fit(X_train, y_train)


# Predictions 
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)


# Model Building --> K-Nearest Neighbors
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto')
model.fit(X_train, y_train)


# Model Building --> K-Nearest Neighbors
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto')
model.fit(X_train, y_train)


# Model Building --> K-Nearest Neighbors
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='ball_tree')
model.fit(X_train, y_train)


# Model Building --> K-Nearest Neighbors
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='brute')
model.fit(X_train, y_train)


# Model Building --> LightGBM
from lightgbm import LGBMRegressor
model = LGBMRegressor(n_estimators=100, learning_rate=0.1, boosting_type='gbdt')
model.fit(X_train, y_train)


# Model Building --> Decision Tree
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(criterion='squared_error', splitter='best', min_samples_split=2)
model.fit(X_train, y_train)


# Evaluation - Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy Score on Train Set: ", accuracy_score(y_train, y_pred_train))
print("Accuracy Score on Test Set: ", accuracy_score(y_test, y_pred_test))


# Evaluation - Precision
from sklearn.metrics import precision_score
print("Precision Score on Train Set: ", precision_score(y_train, y_pred_train))
print("Precision Score on Test Set: ", precision_score(y_test, y_pred_test))


# Evaluation - AUC Score
from sklearn.metrics import roc_auc_score
print("AUC Score on Train Set: ", roc_auc_score(y_train, y_pred_train))
print("AUC Score on Test Set: ", roc_auc_score(y_test, y_pred_test))