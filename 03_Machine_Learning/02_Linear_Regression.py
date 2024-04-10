# Linear Regression
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt


# # Create a simple dataset with 100 samples
# X = np.linspace(-1, 1, 100)
# # Create a non-linear relationship between X and y
# y = 2 * X ** 2 + np.random.randn(*X.shape) * 0.33


# # Create a simple Linear Regression Model
# model = LinearRegression()
# model.fit(X.reshape(-1, 1), y.reshape(-1, 1))

# # Graph the dataset and the line of best fit
# plt.scatter(X, y)
# plt.plot(X, model.predict(X.reshape(-1, 1)), color='red')
# plt.title('Linear Regression')
# plt.savefig('images/02-1.png')
# plt.clf()

# Iris Dataset
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pandas import DataFrame as df

# Load the Iris dataset
def linearregression_iris():
    iris = datasets.load_iris()
    print(iris.data.shape)
    print(iris.feature_names)
    print(iris.target_names)
    # Turn into a dataframe
    iris_df = df(data=iris.data, columns=iris.feature_names)
    print(iris_df.head())

    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=42)

    # Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)  # R^2
    test_score = model.score(X_test, y_test)
    print('Linear Regression Train Score: ', train_score)
    print('Linear Regression Test Score: ', test_score)

    predictions = model.predict(X_test)
    print('Linear Regression Accuracy: ', accuracy_score(y_test, predictions.round()))

    # Coefficient and Intercept
    print('Linear Regression Coefficient: ', model.coef_)
    print('Linear Regression Intercept: ', model.intercept_)

    # Graph the dataset 
    plt.scatter(X_train[:, 0], y_train, color='blue')

    # Plot the regression line
    plt.plot(X_train[:, 0], model.predict(X_train), color='red')
    plt.savefig('images/02-2.png')


# Load the Diabetes dataset
def linearregression_diabetes():
    diabetes = datasets.load_diabetes()
    print(diabetes.data.shape)
    print(diabetes.feature_names)
    print(diabetes.target.shape)
    # Turn into a dataframe
    diabetes_df = df(data=diabetes.data, columns=diabetes.feature_names) 
    diabetes_df['target'] = diabetes.target
    print(diabetes_df.head())

    # Split the dataset into training and testing sets
    # Use only three features 'age', 'sex'
    X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.33, random_state=42)

    # Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)  # R^2
    test_score = model.score(X_test, y_test)
    print('Linear Regression Train Score: ', train_score)
    print('Linear Regression Test Score: ', test_score)

    # Coefficient and Intercept
    print('Linear Regression Coefficient: ', model.coef_)
    print('Linear Regression Intercept: ', model.intercept_)

    # Graph the dataset 
    plt.scatter(X_train[:, 0], y_train, color='blue')
    

    # Plot the regression line
    plt.plot(X_train[:, 0], model.predict(X_train), color='red')
    plt.savefig('images/02-3.png')

linearregression_diabetes()