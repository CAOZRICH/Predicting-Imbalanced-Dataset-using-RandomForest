#credit-card-fraud-detection-by#caozrich

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from help_functions import is_cat
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

    
class creditCardFraudDetector():
    
    def __init__(self,odf,y,reduce = False) :
        self.df     = odf.copy()
        self.y      = y  #independent variable name from the dataset
        self.reduce = reduce
        X_train_prep, X_test_prep,y_train,y_test = self.preprocessing(df,y,reduce)
        self.trainAlgorithm(X_train_prep, X_test_prep,y_train,y_test)
        
    def preprocessing(self,df,y,reduce):
        if reduce == True:
           df = self.reduceDatasetlenthg(df,y) #find the most correlated columns to reduce the length of the data frame 
        df = self.deleteinvalidData(df)  #checks the data set for null values or inf/-inf values and deletes them     
        X_train, X_test, y_train, y_test = train_test_split(df, df[y],shuffle=True, test_size=0.33, random_state=42, stratify=df[y])
        X_train, X_test,y_train,y_test = self.transformCatTonum(X_train, X_test,y_train,y_test) #transfrom categorical data to numeric        
        X_train_prep,X_test_prep         = self.fillNanValue(X_train, X_test,y_train,y_test) 
        return X_train_prep, X_test_prep,y_train,y_test 

        
    def reduceDatasetlenthg(self,df,y): 
        threshold = 0.2 #correlation level for filtering
        df2 = df.copy()        
        if is_cat(df[y]) == True:                
           df2[y] = df2[y].astype('category').cat.codes
        sries    = df2.corr().loc[y].sort_values(ascending=False) 
        sries    = sries.iloc[1:]
        rslt_df  = sries.loc[sries > threshold]
        lista    = rslt_df.index.tolist()
        lista.append(y)
        df       = df[lista]
        return df 
            
    def deleteinvalidData(self,df): 
        for i in df.columns[df.isin([np.inf, -np.inf]).any()].tolist():
            df = df.drop(i, axis=1)
            
        return df    
    
    def fillNanValue(self, X_train, X_test, y_train, y_test):
        imputer = SimpleImputer(strategy="median")
        
        X_train_prep = imputer.fit_transform(X_train)
        X_test_prep  = imputer.fit_transform(X_test)
        
        X_train_prep = pd.DataFrame(X_train_prep, columns=X_train.columns, index=y_train.index)
        X_test_prep  = pd.DataFrame(X_test_prep, columns=X_test.columns, index=y_test.index)
        return X_train_prep,X_test_prep
                
        
    def transformCatTonum(self,X_train, X_test, y_train , y_test):       
        if is_cat(y_train) == True:         
            y_train = y_train.astype('category').cat.codes
            y_test  = y_test.astype('category').cat.codes     
        for col in X_train.columns:
           if is_cat(X_train[col]) == True:
                X_train[col]   = X_train[col].astype('category').cat.codes 
                X_test[col]    = X_test[col].astype('category').cat.codes  
        return X_train, X_test, y_train  , y_test   
    
    def trainAlgorithm(self, X_train_prep , X_test_prep, y_train, y_test):
        X_train_prep = X_train_prep.drop(columns=['is_fraud'])
        X_test_prep = X_test_prep.drop(columns=['is_fraud'])
        
        clf_tree = DecisionTreeClassifier(random_state=42)
        clf_tree.fit(X_train_prep, y_train)
        
        y_pred  = clf_tree.predict(X_test_prep)
        y_predps =  pd.Series(y_pred) 
        print(y_predps.value_counts())
        print("F1 Score:", f1_score(y_pred, y_test))
        cm = confusion_matrix(y_pred, y_test)
        ax= plt.subplot()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax);  

        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
        ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(['Not Fraud', 'Fraud']); ax.yaxis.set_ticklabels(['Not Fraud', 'Fraud']);
        plt.show()
                        
      
df = pd.read_csv("fraudTrain.csv")  
cdf = creditCardFraudDetector(df,'is_fraud',reduce = True) 
