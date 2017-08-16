import pandas
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

# Load the data set - must be one-hot encoded data for now
df = pandas.read_csv("ml_data_set.csv")

y = df['result'].as_matrix()

del df['result']

X = df.as_matrix()

# Split the data set 70-30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = ensemble.GradientBoostingRegressor()

# Parameters to try
param_grid = {
    'n_estimators': [1000, 3000],
    'max_depth': [4, 6],
    'min_samples_leaf': [3, 5, 17],
    'learning_rate': [0.1, 0.05, 0.01],
    'max_features': [1.0, 0.3, 0.1],
    'loss': ['lad', 'huber']
}

# Define the grid search - n_jobs is the number of usable cpu cores
gs_cv = GridSearchCV(model, param_grid, n_jobs=4)

# Run the grid search on training data
gs_cv.fit(X_train, y_train)

# Print the the best parameters
print(gs_cv.best_params_)

# Find the error rate on the training set using the best parameters
mse = mean_absolute_error(y_train, gs_cv.predict(X_train))
print("Training - Absolute Error: %.4f" % mse)

# Find the error rate on the test set using the best parameters
mse = mean_absolute_error(y_test, gs_cv.predict(X_test))
print("Test - Absolute Error: %.4f" % mse)

