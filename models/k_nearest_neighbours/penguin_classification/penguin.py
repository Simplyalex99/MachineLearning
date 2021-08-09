import pandas as pd
import numpy as np
import sklearn
from sklearn import  preprocessing
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def print_predictions(model,x_test,y_test):
    predictions = model.predict(x_test)
    for i in range(len(predictions)):
        print(predictions[i],x_test[i],y_test[i])

def k_nearest_neighbours(X,y):
    x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.10)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train,y_train)
    accuracy = model.score(x_test,y_test)
    print(accuracy)
def plot_data(x,y,plot_labels_dict):
    plt.scatter(x,y)
    if(plot_labels_dict):
        plt.xlabel(plot_labels_dict.get('xlabel'))
        plt.ylabel(plot_labels_dict.get('ylabel'))
        plt.title(plot_labels_dict.get('title'))
    plt.show()
def natural_encoding(dataframe):
    le = preprocessing.LabelEncoder()
    for column in dataframe.columns:
        dataframe[column] = le.fit_transform(dataframe[column])
        dataframe
    return dataframe
def read_csv(file,**kwargs):
    dataframe = pd.read_csv(file,**kwargs)
    return dataframe
def print_distribution(x,**kwargs):

    # Sturge’s rule; formula for number of bins:
    # K = 1 + 3.322 log(N) , where N = floor(max_data_point)-floor(min_data_point)
    plt.hist(x, color='blue', edgecolor='black',
             bins=int(7))
    plt.show()

def main():
    penguin_df = read_csv('iris.csv')
    columns = ["Iris-versicolor","Iris-setosa","Iris-virginica"]
    penguin_df = natural_encoding(penguin_df)
    print(penguin_df.head())
    X = np.array(penguin_df.drop(['class'],1),dtype=int)
    y = np.array(penguin_df['class'],dtype=int)
    k_nearest_neighbours(X,y)


def get_user_input():
    user_option = input("Type 'Y' to predict a penguin on atttributes of your choice.\nOr type 'N' to go with default.")
    input_values = {}
    if(user_option.upper() == 'Y'):
        label = input("Please type  the attribute you want to predict from the following attributes:\nsepal_length_cm"
                      "sepal_width_cm petal_length_cm petal_width_cm class")
        columns = ["sepal_width_cm","petal_length_cm","petal_width_cm","class"]
        for column in columns:
            if column != label:
                input_value =  input("Please enter value for {}").format(column)
                input_values[column] = input_value
    return (label, input_values) 
    

