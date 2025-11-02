import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


def make_boxplot(data, attribute, by=None):
    if by is not None:
        boxplot = data.plot.box(column=attribute, by=by, figsize=(10, 8))
    else:
        boxplot = data.plot.box(column=attribute, figsize=(10, 8))
    plt.show()


def make_histogram(attribute):
    histogram = attribute.hist()
    histogram.set_xlabel(attribute.name)
    histogram.set_ylabel("Frequency")
    plt.show()


def make_scatter_plot(data, x_attribute, y_attribute):
    scatter_plot = data.plot.scatter(x=x_attribute.name, y=y_attribute.name)
    plt.show()


def calculate_and_show_numerical_data(data, attribute, by=None):
    mean = attribute.mean()
    median = attribute.median()
    mode = attribute.mode().values
    range = attribute.max() - attribute.min()
    std = attribute.std()
    variance = attribute.var()
    quartiles = [attribute.quantile(0.25), attribute.quantile(0.5), attribute.quantile(0.75)]
    make_boxplot(data, attribute.name)
    if by is not None:
        make_boxplot(data, attribute.name, by)
    make_histogram(attribute)

    print(attribute.name)
    print("Mean: ", mean)
    print("Median: ", median)
    print("Mode: ", mode)
    print("Range: ", range)
    print("Standard Deviation: ", std)
    print("Variance: ", variance)
    print("Quartiles: ", quartiles)
    print("\n")


data = pd.read_csv("cng514-covid-survey-data.csv")
data_orginal = data.copy()
data["Gender"] = data["Gender"].astype('str')
income = data.AnnualHouseholdIncome
mask_intent = data.CoronavirusIntent_Mask
gender = data.Gender

# Annual Household Income
calculate_and_show_numerical_data(data, income, gender.name)

# Coronavirus Intent Mask
calculate_and_show_numerical_data(data, mask_intent, gender.name)

# Gender
data_encoded = data.copy()
data_encoded["Gender"] = data_encoded["Gender"].astype('category')
data_encoded["Gender"] = data_encoded["Gender"].cat.codes
gender_labeled = data_encoded.Gender
mode = gender.mode().values
make_boxplot(data_encoded, gender_labeled.name)
make_histogram(gender)

print("Gender")
print("Mode: ", mode)
print("Values: ", gender.value_counts())
print("Values: ", gender_labeled.value_counts())
print("\n")

make_scatter_plot(data, income, mask_intent)
make_scatter_plot(data, income, gender)
make_scatter_plot(data, mask_intent, gender)
plt.show()

# Missing values
print("Number of missing Income values: ", income.isnull().sum())
print("Number of missing Income values: ", mask_intent.isnull().sum())
print("Number of missing Income values: ", data_orginal.Gender.isnull().sum())
print("\n")

# Normalization
scaler = preprocessing.MinMaxScaler()
data_normalized_min_max = data.copy()
data_normalized_min_max["AnnualHouseholdIncome"] = scaler.fit_transform(
    data_normalized_min_max[["AnnualHouseholdIncome"]])
data_normalized_min_max["CoronavirusIntent_Mask"] = scaler.fit_transform(
    data_normalized_min_max[["CoronavirusIntent_Mask"]])
print(data_normalized_min_max)

scaler = preprocessing.StandardScaler()
data_normalized_z_score = data.copy()
data_normalized_z_score["AnnualHouseholdIncome"] = scaler.fit_transform(
    data_normalized_z_score[["AnnualHouseholdIncome"]])
data_normalized_z_score["CoronavirusIntent_Mask"] = scaler.fit_transform(
    data_normalized_z_score[["CoronavirusIntent_Mask"]])
print(data_normalized_z_score)

# Correlation
correlation = income.corr(mask_intent)
print("Correlation between Income and Mask Intent: ", correlation)
correlation = income.corr(gender_labeled)
print("Correlation between Income and Gender: ", correlation)
correlation = mask_intent.corr(gender_labeled)
print("Correlation between Mask Intent and Gender: ", correlation)
