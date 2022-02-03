""" 
In order to test the data, it is necessary to install pytest package by:
pip install -U pytest

To run testing in terminal using:
python -m pytest customized_packages/unit_test
"""

from customized_packages.data import get_bank_data
import pandas as pd

numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']

def test_bank_marketing_data():
    """Function to examine the output of function to download data"""
    
    # download data
    data = get_bank_data()
    
    print("\nTesting data...!")
    # Check if all columns of download data in corrected form
    assert all(data.columns == ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
           'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
           'previous', 'poutcome', 'y'])
    
    # Check data shape
    assert data.shape == ((4521, 17))
    
    # Check if output variable is binary
    assert data["y"].nunique() == 2
    
    # Check if numerical columns are in correct form
    for numerical_column in numerical_columns:
        assert data[numerical_column].dtype == "int"
    
    # Check if categorical columns are in correct form
    for categorical_colum in categorical_columns:
        assert data[categorical_colum].dtype == "object"
    
    print("Done!")