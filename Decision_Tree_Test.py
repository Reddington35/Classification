import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree,metrics

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

# array that stores gini and entropy as strings & min_sample_value as a list in range 1 - 10,
# two separate empty arrays for training accuracy and test accuracy
criterion_type = ["gini","entropy"]
min_sample_value = list(range(1, 10))
array_acc_training = []
array_acc_test = []

# nested loop, first loop goes through the length of criterion_type array and appends the training data from that array
# to inner array here, same for test data, next loop goes from 1 -10 training the data at each interval
# with these two hyper-parameters note: i and x represents their position in the loop then prints the results out
for i in range(len(criterion_type)):
    array_acc_training.append([])
    array_acc_test.append([])

    for x in min_sample_value:
        model = tree.DecisionTreeClassifier(criterion=criterion_type[i],min_samples_leaf=x)
        model.fit(x_training, y_training)

        # Evaluates all the training & Test Data
        predictions_training = model.predict(x_training)
        predictions_test = model.predict(x_test)

        # Evaluates all the accuracy of the training and test predictions
        array_acc_training[i].append(metrics.accuracy_score(y_training, predictions_training))
        array_acc_test[i].append(metrics.accuracy_score(y_test, predictions_test))
print("Accuracy on training data:",array_acc_training)
print("Accuracy on test data:",array_acc_test)

# plots test data using matplotlib library gini is at position 0 , entropy is at position 1
# note: value of min_samples_value is on the x axis, accuracy is on the y axis
plt.scatter(min_sample_value,array_acc_test[0],marker="x")
plt.scatter(min_sample_value,array_acc_test[1],marker="*")
plt.xlim([0, max(min_sample_value)+2])
plt.ylim([0.0, 1.1])
plt.xlabel("min samples leaf value")
plt.ylabel("Accuracy")
legend_labels = ["gini","entropy"]
plt.legend(labels=legend_labels, loc=4, borderpad=1)
plt.title("Effect of Min Leaf Samples and Criterion Hyper-Perameters on test accuracy", fontsize=10)
plt.show()

# References
# Mannion, P., 2021. .