import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# variables that contain .csv files
training_file = "wildfires_training.csv"
test_file = "wildfires_test.csv"

# arrays of String's that contain independent data and target data
independant_columns = ["year","temp","humidity","rainfall","drought_code","buildup_index","day","month","wind_speed"]
dependant_column = ["fire"]

# variable for reading the data from the training_file, and print contents of variable
df_training = pd.read_csv(training_file)
#print(df_training)

# seperator for columns in the independent variables dataframe and prints out
x_training = df_training.loc[:,independant_columns]
#print(x_training)

# seperator for columns in the target variable dataframe and prints out
y_training = df_training.loc[:,dependant_column]
#print(y_training)

# reads test_file and stores contents to variable then prints contents
df_test = pd.read_csv(test_file)
#print(df_test)

# seperator for columns in the independent variables dataframe and prints out
x_test = df_test.loc[:,independant_columns]
#print(x_test)

# seperator for columns in the target variable dataframe and prints out
y_test = df_test.loc[:,dependant_column]
#print(y_test)

# classifier being used is LogisticRegressionClassifier,this program is just to analyse the data without the
# tuning of hyper-parameters, default hyper-perimeter for solver is lbfgs however would not work for this model as
# max_iter default value is 100 therefore I increased the max_iter value to solve this issue
# note: y_training.values.ravel() is needed to turn the target data in y_training into a 1d array for the purposes of
# this model
model = LogisticRegression(max_iter=200)
model.fit(x_training, y_training.values.ravel())

# prediction data for training and test data
predictions_training = model.predict(x_training)
predictions_test = model.predict(x_test)

# accuracy of training and test data with respect to the model being used
print("Training Data: ",metrics.accuracy_score(y_training, predictions_training))
print("Test Data: ",metrics.accuracy_score(y_test, predictions_test))

# References
# Mannion, P., 2021. .

