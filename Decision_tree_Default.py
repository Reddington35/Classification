import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree
from sklearn import tree,metrics

# variables that contain .csv files
training_file = "wildfires_training.csv"
test_file = "wildfires_test.csv"

# arrays of String's that contain independent data and target data
independant_columns = ["year","temp","humidity","rainfall","drought_code","buildup_index","day","month","wind_speed"]
dependant_column = ["fire"]

# variable for reading the data from the training_file, and print contents of variable
df_training = pd.read_csv(training_file)
print(df_training)

# seperator for columns in the independent variables dataframe and prints out
x_training = df_training.loc[:,independant_columns]
print(x_training)

# seperator for columns in the target variable dataframe and prints out
y_training = df_training.loc[:,dependant_column]
print(y_training)

# reads test_file and stores contents to variable then prints contents
df_test = pd.read_csv(test_file)
print(df_test)

# seperator for columns in the independent variables dataframe and prints out
x_test = df_test.loc[:,independant_columns]
print(x_test)

# seperator for columns in the target variable dataframe and prints out
y_test = df_test.loc[:,dependant_column]
#print(y_test)

# classifier being used is DecisionTreeClassifier,this program is just to analyse the data without the
# use of hyper-parameters, default hyper-perimeter is gini
model = tree.DecisionTreeClassifier()
model.fit(x_training, y_training)

# prediction data for training and test
predictions_training = model.predict(x_training)
predictions_test = model.predict(x_test)

# accuracy of training and test data with respect to the model being used
print("Training Data:" ,metrics.accuracy_score(y_training, predictions_training))
print("Test Data: " , metrics.accuracy_score(y_test, predictions_test))

# using matplotlib plotted out a tree with variable labels for a visualisation of the tree without any iterations
sklearn.tree.plot_tree(model,feature_names = independant_columns,filled=True,label = "all")
plt.show()

# References
# Mannion, P., 2021. .
