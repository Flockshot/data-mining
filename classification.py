import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def evaluate_model(data_x, data_y, k):
    k_fold = KFold(10, shuffle=True, random_state=1)

    predicted_targets = np.array([])
    actual_targets = np.array([])

    for train_ix, test_ix in k_fold.split(data_x):
        train_x, train_y, test_x, test_y = data_x[train_ix], data_y[train_ix], data_x[test_ix], data_y[test_ix]

        # Standardize the data
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)

        # Fit the classifier
        knn = KNeighborsClassifier(n_neighbors=k)
        classifier = knn.fit(train_x, train_y)

        # Predict the labels of the test set samples
        predicted_labels = classifier.predict(test_x)

        predicted_targets = np.append(predicted_targets, predicted_labels)
        actual_targets = np.append(actual_targets, test_y)

    return predicted_targets, actual_targets


def plot_confusion_matrix(cnf_matrix):
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    generate_confusion_matrix(cnf_matrix, classes=["No", "Yes"], title='Confusion matrix, without normalization')
    plt.show()

    # Plot normalized confusion matrix
    plt.figure()
    generate_confusion_matrix(cnf_matrix, classes=["No", "Yes"], normalize=True, title='Normalized confusion matrix')
    plt.show()


def generate_confusion_matrix(cnf_matrix, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.

    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return cnf_matrix


data_orig = pd.read_csv("classification-data.csv")
data = pd.read_csv("classification-data.csv")

# Converting the categorical data to numerical data.
le = preprocessing.LabelEncoder()
# 1 is Male, 0 is Female
data['GENDER'] = le.fit_transform(data['GENDER'])
# 1 is Yes, 0 is No
data['LUNG_CANCER'] = le.fit_transform(data['LUNG_CANCER'])


# Checking for missing data.
for column in data.columns:
    if data[column].isnull().sum() > 0:
        print(column, "has missing values")

    # Check if data is an integer.
    if data[column].dtype == 'int64' or data[column].dtype == 'int32':
        # print max value in the column
        print(column, "max value is ", data[column].max(), "and min value is ", data[column].min())


X = data.drop('LUNG_CANCER', axis=1)
y = data['LUNG_CANCER']

k_values = [i for i in range(1, 31)]
acc_scores = []

for k in k_values:
    predicted_targets, actual_targets = evaluate_model(X.values, y.values, k)
    acc_scores.append(accuracy_score(actual_targets, predicted_targets))

error = np.array(1) - np.array(acc_scores)

sns.lineplot(x=k_values, y=error, marker='o')
plt.xlabel("K Values")
plt.ylabel("Error")
plt.show()

# Using elbow method I determine k=5 to be the best value for k.
best_k = 5
predicted_target, actual_target = evaluate_model(X.values, y.values, best_k)
cnf_matrix = confusion_matrix(actual_target, predicted_target)
plot_confusion_matrix(cnf_matrix)

tn, fp, fn, tp = cnf_matrix.ravel()

print("Specificity: ", tn / (tn + fp))
print("Sensitivity: ", recall_score(actual_target, predicted_target))
print("Accuracy: ", accuracy_score(actual_target, predicted_target))
print("F1 Score: ", f1_score(actual_target, predicted_target))
print("Precision: ", precision_score(actual_target, predicted_target))






# scores = ["accuracy", "precision", "recall", "f1"]
#
# for k in k_values:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     cv_results = cross_validate(knn, X, y, scoring= cv=10)
#
#
#     scores.append(np.mean(score))
#
# sns.lineplot(x=k_values, y=scores, marker='o')
# plt.xlabel("K Values")
# plt.ylabel("Accuracy Score")
# plt.show()

# # Splitting the data into training and testing data.
# X = data.iloc[:, [x for x in range(data.columns.size - 2)]]
# Y = ravel(data.iloc[:, [data.columns.size - 1]])
# X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, random_state=0)
#
# clf = Pipeline(
#     steps=[("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=3))]
# )
#
# clf.fit(X_train, y_train)
# print(clf.score(X_test, y_test))


# x_train, x_test, y_train, y_test = cross_validation.train_test_split(data,labels, test_size=0.4, random_state=1 )
# knn = KNeighborsClassifier(n_neighbors=3)


# X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
#
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
#
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, Y_train)
#
# Y_pred = knn.predict(X_test)
# accuracy = accuracy_score(Y_test, Y_pred)
# print("Accuracy:", accuracy)
