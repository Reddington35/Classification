import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# arrays of String's that contain independent data and target data
training_file = "wildfires_training.csv"
test_file = "wildfires_test.csv"

# variable for reading the data from the training_file, and print contents of variable
independant_columns = ["year","temp","humidity","rainfall","drought_code","buildup_index","day","month","wind_speed"]
dependant_column = ["fire"]

# seperator for columns in the independent features dataframe and prints out
df_training = pd.read_csv(training_file)
#print(df_training)

# seperator for columns in the target feature dataframe and prints out
x_training = df_training.loc[:,independant_columns]
#print(x_training)

# reads test_file and stores contents to variable then prints contents
y_training = df_training.loc[:,dependant_column]
#print(y_training)

# reads test_file and stores contents to variable then prints contents
df_test = pd.read_csv(test_file)
#print(df_test)

# seperator for columns in the independent features dataframe and prints out
x_test = df_test.loc[:,independant_columns]
#print(x_test)

# seperator for columns in the target feature dataframe and prints out
y_test = df_test.loc[:,dependant_column]
#print(y_test)

# array that stores l1 and l2 as strings & min_sample_value as a list in range 1 - 10,
# two separate empty arrays for training accuracy and test accuracy
penalty_type = ["l1","l2"]
C_type = list(range(1, 10))
array_acc_training = []
array_acc_test = []

# nested loop, first loop goes through the length of penalty_type array and appends the training data from that array
# to inner array here, same for test data, next loop goes from 1 -10 training the data at each interval
# with these two hyper-parameters note: i and x represents their position in the loop then prints the results out
for i in range(len(penalty_type)):
    array_acc_training.append([])
    array_acc_test.append([])

    for x in C_type:
        model = LogisticRegression(penalty=penalty_type[i],C=x,solver="liblinear")
        model.fit(x_training, y_training.values.ravel())

        # Evaluates all the training & Test Data
        predictions_training = model.predict(x_training)
        predictions_test = model.predict(x_test)

        # Evaluates all the accuracy of the training and test predictions
        array_acc_training[i].append(metrics.accuracy_score(y_training, predictions_training))
        array_acc_test[i].append(metrics.accuracy_score(y_test, predictions_test))
print("Accuracy on training data:",array_acc_training)
print("Accuracy on test data:",array_acc_test)

# plots training data using matplotlib library l1 is at position 0 , l2 is at position 1
# note: value of C is on the x axis, accuracy is on the y axis
plt.scatter(C_type, array_acc_training[0], marker="x")
plt.scatter(C_type, array_acc_training[1], marker="*")
plt.xlim([0, max(C_type) + 2])
plt.ylim([0.0, 1.1])
plt.xlabel("Value of C")
plt.ylabel("Accuracy")
legend_labels = ["Penalty l1 using liblinear","penalty l2 using liblinear"]
plt.legend(labels=legend_labels, loc=4, borderpad=1)
plt.title("Effect of C,Liblinear & Penalty Hyper-Perameters on training accuracy", fontsize=10)
plt.show()

# Mannion, P., 2021. .




