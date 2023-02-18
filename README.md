# Treatment of an imbalanced dataset and use of Random Forest to classify it

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

## Imbalance Visualization 

We are going to see how big the imbalance is in the "is_fraud" class of the dataset (There can also be imbalances in dependent variables and can be corrected using some scaling technique).


<p align="left">
  <img src="https://user-images.githubusercontent.com/34092193/215362440-8d0106d8-5e20-439e-a4d0-d5e8bc89cc79.png"/>
</p>

## Imbalance Reducction 

Now we are going to create a sample with the same proportions of normal transactions and anomaly transactions using the .sample method

```python

normal = df[df[y] == 0]
fraud  = df[df[y] == 1]
Sample = normal.sample(n=7506)¿
sampledData = pd.concat([Sample, fraud], axis = 0)	
```

# Results

<p align="left">
  <img src="https://user-images.githubusercontent.com/34092193/219877340-c4c8fb9f-c977-4dfc-8a47-9d482d509c02.png"/>
</p>


## Reduction of dataset size

In most datasets, there can be data that does not relate to the class to be predicted, so we have the option of reducing the size of the dataset to obtain some improvement in the time of preprocessing and the use of machine learning algorithms. The following image shows the correlation between the variables in the dataset:

<p align="left">
  <img src="https://user-images.githubusercontent.com/34092193/215363492-3af4f911-c8e5-4865-9dea-c43441d9676f.png"/>
</p>

As we can see, there is only one variable correlated with the independent class, which is the 'atm' column, so in this case, we can dispense with the others. In the same way, a comparison of the results obtained between the entire dataset and the reduced dataset will be made at the end.

## Results

<p align="left">
  <img src="https://user-images.githubusercontent.com/34092193/219877340-c4c8fb9f-c977-4dfc-8a47-9d482d509c02.png"/>
</p>

After applying preprocessing to optimize the results obtainable through the use of the random forest algorithm, we will analyze the results using the confusion matrix metric.

<p align="left">
  <img src="https://user-images.githubusercontent.com/34092193/215363976-4bc9d4dd-0682-4bd4-b119-c673c777a825.png"/>
</p>

We can see that the random forest with this dataset performed very well, achieving 0 prediction errors, this indicates how well this algorithm can perform with this type of problem, other algorithms that are quite good with imbalanced datasets are Decision Tree, Adaboost, and Gradient Boosting. Now let's compare the results obtained from the reduced dataset with the full dataset.

### F1 score

|Reduced Dataset|Full dataset|
|----|----|
|1.0|1.0|

In this case, there is no difference between the two as we saw in the graphical representation of correlations, only one variable had correlation with the class to predict, so it was right to apply a method of reducing the size of the dataset.

## References

- [1] Credit Card Transactions Fraud Detection Dataset https://www.kaggle.com/datasets/kartik2112/fraud-detection
- [2] Sparkov_Data_Generation https://github.com/namebrandon/Sparkov_Data_Generation



