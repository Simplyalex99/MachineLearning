import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

import matplotlib.pyplot as plt

# @PARAMS: 
# @linear linear regression object,
# @x_test test input sample, 
# @y_test test output sample

def print_predictions(linear,x_test,y_test):
    predictions = linear.predict(x_test)
    for i in range(len(predictions)):
        print(predictions[i],x_test[i],y_test[i])

# @PARAMS: 
# @x list type containning inputs,
# @y list type containning output , 
# @plot_labels_dict dictionary containning information about the plot labels
# @**kwargs features for scatter plot
def plot_2d_data(x,y,plot_labels_dict,**kwargs):
    plt.scatter(x,y,**kwargs)
    if(plot_labels_dict):
        plt.xlabel(plot_labels_dict.get('xlabel'))
        plt.ylabel(plot_labels_dict.get('ylabel'))
        plt.title(plot_labels_dict.get('title'))
    plt.show()

# @PARAMS: 
# @x array type containning inputs,
# @y array type containning output, 

def train_model(X,y):
    x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.33)
    linear = linear_model.LinearRegression()
    linear.fit(x_train,y_train)
    accuracy = linear.score(x_test,y_test)
    print('accuracy of model: ',accuracy)

    # test predictions
    print_predictions(linear, x_test, y_test)

# @PARAMS: 
# @file csv file to be read
# @**kwargs features for reading the csv file, 

def read_csv(file, **kwargs):
  data = pd.read_csv(file, **kwargs)
  return data

def main():
    # collects data
    header_options = {'sep':","}
    salary_dataframe = read_csv('Salary.csv',**header_options)

    # visualize data to check linear relationship
    x_axis = list(salary_dataframe['YearsExperience'])
    y_axis = list(salary_dataframe['Salary'])
    plot_labels_dict = {'xlabel':'Years Of Experience','ylabel':'Salary $/hr','title':'Salary Yearly Wage'}
    plot_2d_data(x_axis,y_axis,plot_labels_dict)

    # creates model for single variable relationship using reshape
    X = np.array(salary_dataframe[["YearsExperience"]])
    y = np.array(salary_dataframe['Salary'])
    train_model(X.reshape(-1,1),y.reshape(-1,1))


if __name__ == '__main__':
    main()