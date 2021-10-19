import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import neural_network
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# no more warnings!
import warnings

warnings.filterwarnings('ignore')

# load the file - IRIS
# data_file = "IRIS.csv"

# load the file - camera
data_file = "camera_dataset.csv"
# read the file
dataset = pd.read_csv(data_file)
# dataset = np.reshape((150, 1))


# only use the wanted features
features = dataset[["Max resolution", "Low resolution", "Effective pixels", "Zoom wide (W)",
                    "Zoom tele (T)", "Normal focus range", "Macro focus range", "Storage included",
                    "Weight (inc. batteries)", "Dimensions", "Price"]]

# IRIS.csv Features
# features = dataset[["sepal_length", "sepal_width", "petal_length", "petal_width"]]

labels = dataset[["Release date"]]
# IRIS.csv Labels
# labels = dataset[["species"]]

features_training_data, features_testing_data, labels_training_data, labels_testing_data = \
    train_test_split(features, labels, test_size=.50)

# training below
classifier_decision_tree = tree.DecisionTreeClassifier()
classifier_neural_network = neural_network.MLPClassifier()
classifier_k_nearest_neighbors = KNeighborsClassifier()
linear_regression = linear_model.LinearRegression()

# features_training_data.fillna(features_training_data.mean(), inplace=True)
# features_testing_data.fillna(features_testing_data.mean(), inplace=True)
# labels_training_data.fillna(labels_training_data.mean(), inplace=True)
# labels_testing_data.fillna(labels_testing_data.mean(), inplace=True)


classifier_decision_tree = classifier_decision_tree.fit(features_training_data, labels_training_data)
classifier_neural_network = classifier_neural_network.fit(features_training_data, np.ravel(labels_training_data))
classifier_k_nearest_neighbors = classifier_k_nearest_neighbors.fit(features_training_data,
                                                                    np.ravel(labels_training_data))

linear_regression = linear_regression.fit(features_training_data, labels_training_data)
# linear_regression = linear_regression.fit(features_training_data, features_testing_data)


# Now to start predicting
predicitons_for_cdt = classifier_decision_tree.predict(features_testing_data)
predicitons_for_cnn = classifier_neural_network.predict(features_testing_data)
predicitons_for_cknn = classifier_k_nearest_neighbors.predict(features_testing_data)
predicitons_for_lr = linear_regression.predict(features_testing_data)

# Display info to users
percent_score_cdt = accuracy_score(labels_testing_data, predicitons_for_cdt)
percent_cdt = ("{:.0%}".format(percent_score_cdt))
print("Decision Tree Classifier prediction score is: ", percent_cdt)

percent_score_cnn = accuracy_score(labels_testing_data, predicitons_for_cnn)
percent_cnn = ("{:.0%}".format(percent_score_cnn))
print("Neural Network Classifier prediction score is: ", percent_cnn)

percent_score_cknn = accuracy_score(labels_testing_data, predicitons_for_cknn)
percent_cknn = ("{:.0%}".format(percent_score_cknn))
print("K Nearest Neighbors prediction score is: ", percent_cknn)

percent_score_lr = accuracy_score(labels_testing_data, predicitons_for_lr.round())
percent_lr = ("{:.0%}".format(percent_score_lr))
print("Linear Regression prediction score is: ", percent_lr)

print("\n\n\t **** END OF PROGRAM ****\n\n\n")
