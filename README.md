# Predicting Imbalanced Data using Random Forest
* Author: Richard_Libreros

## Software Environment
* Python 
* numpy 
* matplotlib 
* scikit-learn 
* pandas 
* matplotlib


## Description
In this example, I build a classification model with the Random Forest method that aims to detect fraud in credit card transactions using a notably imbalanced dataset to test how effective this technique is in dealing with this type of dataset. Here are some reasons why I chose this algorithm over others.

	* It can solve classification and regression problems, and performs decent estimation on both fronts.
	* Its efficiency is particularly Notable in Large Data sets.
    * It has methods for balancing errors in data sets with imbalanced classes.

## Data Summary
This <a href="https://www.kaggle.com/datasets/kartik2112/fraud-detection">[1]</a> is a simulated credit card transaction dataset created by using Sparkov Data Generation tool created by Brandon Harriscontaining<a href="https://github.com/namebrandon/Sparkov_Data_Generation">[2]</a>. The dataset containing legitimate and fraud transactions from the duration 1st Jan 2019 - 31st Dec 2020. It covers credit cards of 1000 customers doing transactions with a pool of 800 merchants. The following picture shows a sample of the data:

<p align="center">
  <img src="https://user-images.githubusercontent.com/34092193/215361096-4dd41353-288d-4bde-8671-0577eedb4aa9.png"/>
</p>
