from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def is_cat(column): #check if the colum is categorical 
    categorical_dtypes = ['object', 'category', 'bool']
    if column.dtype.name in categorical_dtypes:
        return True
    else:
        return False   
    

# def train_val_test_split(df, rstate=42, shuffle=True, stratify=None): #split the dataset into train set /test set
#     strat = df[stratify] if stratify else None
#     train_set, test_set = train_test_split(
#         df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
#     strat = test_set[stratify] if stratify else None
#     val_set, test_set = train_test_split(
#         test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
#     return (train_set, val_set, test_set)


