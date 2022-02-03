import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, roc_auc_score
from scikitplot.metrics import plot_confusion_matrix, plot_roc, plot_precision_recall

numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

full_columns = ['age', 'job', 'marital', 'education', 'default', 'housing','loan', 'contact', 'day', 'month', 'duration', \
                'campaign', 'pdays','previous', 'poutcome', 'balance', 'y']


#####################################################################################################################

# Functions to visualize data

#####################################################################################################################

def plot_bar(data, x_axis = 'y', hue = None, title = None, figure_size = (6,5), label_rotation = False):
    """Visualize using bar plot"""
    fig = plt.figure(figsize= figure_size)
    plt.title(title, fontweight='bold')
    ax = sns.countplot(x= x_axis, data = data , hue = hue)
    size = float(data.shape[0])
    
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 4, '{:1.2f}%'\
                .format(100 * height/size), ha='center')
    
    if label_rotation:
        plt.xticks(rotation = 90)
        

def plot_heatmap(data, figsize = (10,10), annot=True, ylabel_rotation = False):
    """Plot correlation of all atrributes in data"""
    corr = data.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, cmap="YlGn", annot=annot)
    
    if ylabel_rotation:
        plt.yticks(rotation=0)
        

def plot_hists(data, numerical_column_list = numerical_columns, figsize = (16,12), bins = 30):
    """Plot histogram of numerical attributes"""
    
    fig, ax = plt.subplots(figsize = figsize)
    data[numerical_column_list].hist(ax = ax, bins= bins)
    
    

#####################################################################################################################

# Functions to preprocess data

#####################################################################################################################
    
def contruct_missing_values_table(data):
    """Construct table to check missing values"""
    missing_values = data.isnull().sum()
    missing_value_percent = data.isnull().sum() / len(data) * 100
    table = pd.concat([missing_values, missing_value_percent], axis=1)
    missing_values_table = table.rename(columns = {0 : 'missing values', 1 : 'percentage of total values'})
    return missing_values_table 


def display_distribution_of_categorical_values(data):
    """Display the distribution of categorical values"""
    print("="*50)
    for column in data.select_dtypes(include=['object']).columns:
        table = pd.crosstab(index=data[column], columns='percentage of values', normalize=True)
        table.reset_index(inplace = True)
        display(table)
        print("="*50)

def fill_missing_values(data, fill_missing_columns, missing_string = "unknown"):
    """Cheat "unkown" as na values and fill with mode value"""
    for col in fill_missing_columns:
        if missing_string in data[col].unique():
            data[col][data[col] == missing_string] = np.nan
            fill_with_this_value = data[col].mode()[0]
            data[col].fillna(fill_with_this_value, inplace = True)
    return data


def drop_duplicate(data):
    """Remove all the duplicated information"""
    print(f"Data shape before replacing duplicated values: {data.shape}")
    duplicated_removed_data = data.drop_duplicates()
    print(f"Data shape after replacing duplicated values: {duplicated_removed_data.shape}")
    return duplicated_removed_data


# Following functions return the correct column for the data frame
def get_numerical_columns():
    return numerical_columns
def get_categorical_columns():
    return categorical_columns
def get_full_columns():
    return full_columns


def get_feature_names(numerical_transformer, categorical_transformer, numerical_attributes, categorical_attributes, data):
    """Get the feature names after transformation
    
    Parameters
    ----------
    numerical_transformer : numerical transformer
    categorical_transformer: categorical transformer
        attributes to transform to binary
    numerical_attributes: list or str
        Numerical attributes before transformation
    categorical_attributes : list or str
        Categorical attributes before transformation
    
    Output
    ----------
    Feature names after transformation
    """
    table1 = numerical_transformer.fit_transform(data[numerical_attributes])
    table2 = categorical_transformer.fit_transform(data[categorical_attributes])
    
    table = pd.concat([table1, table2],axis = 1)
    
    transformed_column_names = table.columns
    return transformed_column_names



#####################################################################################################################

# Functions to save and load pre-processed data

#####################################################################################################################

def save_prepared_data(save_data_list, feature_names = None, save_column_names = False):
    """Save processed data"""
    
    STORE_PATH = "prepared_data"
    file_names = ["training_data", "testing_data", "training_label", "testing_label"]

    try:
        os.makedirs(STORE_PATH)
    except:
        pass
    
    for index, file_name in enumerate(file_names):
        with open(STORE_PATH + "/" + file_name + ".pkl", "wb") as file:
            pickle.dump(save_data_list[index], file)
    
    if save_column_names:
        with open(STORE_PATH + "/" + "feature_names.pkl", "wb") as file:
            pickle.dump(feature_names, file)
        
        
def load_prepared_data(load_features_name =True):
    """Load processed data"""
    
    STORE_PATH = "prepared_data"
    
    with open(os.path.join(STORE_PATH, "training_data.pkl"), "rb") as f:
        training_data = pickle.load(f)
        
    with open(os.path.join(STORE_PATH, "training_label.pkl"), "rb") as f:
        training_label = pickle.load(f)
    
    with open(os.path.join(STORE_PATH, "testing_data.pkl"), "rb") as f:
        testing_data = pickle.load(f)
        
    with open(os.path.join(STORE_PATH, "testing_label.pkl"), "rb") as f:
        testing_label = pickle.load(f)
    
    if load_features_name:
        with open(os.path.join(STORE_PATH, "feature_names.pkl"), "rb") as f:
            feature_names = pickle.load(f)
        
    return training_data, testing_data, training_label.to_numpy(), testing_label.to_numpy(), feature_names





#####################################################################################################################

# Functions to evaluate model and feature importacances

#####################################################################################################################
def evaluate_model(classifier, X_test, y_test, plot_matrix=True, plot_precision_recall_curve=True):
    
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)
    
    if plot_matrix:
        plot_confusion_matrix(y_test, y_pred)
        plt.show()
    
    if plot_precision_recall_curve:
        plot_precision_recall(y_test, y_prob)
        plt.show()

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("="*40)
    print(f"Precision Score: {precision : .2}")
    print(f"Recall Score: {recall : .2}")
    print("="*40)