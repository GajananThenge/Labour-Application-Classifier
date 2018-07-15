# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 16:16:28 2018

@author: hp-pc
"""

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    print(e)
    
def lable_encoder(data):
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    cat_cols = list(data.select_dtypes(include=['object']).columns)
    for col in cat_cols:
        print(col)
        data[c] = data[c].astype('category')
        data[c] = data[c].cat.codes
    return data


def DataPreprocessing(path):
    
    try:
        #Importing the dataset
        dataset = pd.read_csv(r'I:\Ujjwal\fc62ae9e865b11e8\dataset\train.csv')
        
        dataset['CASE_SUBMITTED_DAY'] = dataset['CASE_SUBMITTED_DAY'].ffill().astype(int)
    
        
    #    dataset['CASE_SUBMITTED_DATE'] = pd.to_datetime(dataset['CASE_SUBMITTED_YEAR'].astype(str) + '-' 
    #           + dataset['CASE_SUBMITTED_MONTH'].astype(str)+'-'+dataset['CASE_SUBMITTED_DAY'].astype(str))
    #    
    #    dataset['DECISION_DATE'] = pd.to_datetime(dataset['DECISION_YEAR'].astype(str) + '-' 
    #           + dataset['DECISION_MONTH'].astype(str)+'-'+dataset['DECISION_DAY'].astype(str))
         
        dataset['Number_Of_Days_to_Process'] = (pd.to_datetime(dataset['DECISION_YEAR'].astype(str) + '-' 
               + dataset['DECISION_MONTH'].astype(str)+'-'+dataset['DECISION_DAY'].astype(str))
        - pd.to_datetime(dataset['CASE_SUBMITTED_YEAR'].astype(str) + '-' 
               + dataset['CASE_SUBMITTED_MONTH'].astype(str)+'-'+dataset['CASE_SUBMITTED_DAY'].astype(str))).dt.days
        
        
        dataset['PREVAILING_WAGE'].fillna(0,inplace=True)
        
    #    dataset.isnull().sum()
    #    
    #    dataset["body_style_cat"] = obj_df["body_style"].cat.codes
        
         feature_cols = ['Number_Of_Days_to_Process','VISA_CLASS',
           'EMPLOYER_COUNTRY', 'SOC_NAME', 'TOTAL_WORKERS',
           'FULL_TIME_POSITION', 'PREVAILING_WAGE', 'PW_UNIT_OF_PAY', 'PW_SOURCE',
           'PW_SOURCE_YEAR', 'PW_SOURCE_OTHER', 'WAGE_RATE_OF_PAY_FROM',
           'WAGE_RATE_OF_PAY_TO', 'WAGE_UNIT_OF_PAY',
            'WORKSITE_STATE', 'WORKSITE_POSTAL_CODE','CASE_STATUS']
         
        dataset = dataset[feature_cols]
        dataset = dataset.dropna()
        
    #    cat_col = list(dataset.select_dtypes(include=['object']).columns)
    #    obj_df = dataset.select_dtypes(include=['object']).copy()
    #    obj_df = obj_df.astype('category')
    #    
    #    for c in cat_col:
    #        obj_df[c] = obj_df[c].cat.codes
    #    
    #    cat_col_index= column_index(dataset,cat_col)
    #    a=dataset[dataset['H-1B_DEPENDENT'].isnull()]
    #    
        
        
        #at_cols = list(dataset.select_dtypes(include=['object']).columns)
        not_num_cols=dataset.select_dtypes(exclude = [np.number,np.int16,np.bool,np.float32] )
        df=dataset
        df_labels = df.copy()
        for col in not_num_cols:
            df_labels[col] = df_labels[col].astype('category')
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes
            
        #Separate independent and dependent data
        X= dataset.iloc[:,:-1].values
        y= dataset.iloc[:,-1].values
        
        #X1=pd.get_dummies(X,dummy_na=False,drop_first=True)
    #    
    #    
    #    
    #    #Encoding the categorial data
    #    
    #    
    #    onehotencoder_X = OneHotEncoder(categorical_features=[0])
    #    X=onehotencoder_X.fit_transform(X).toarray()
    #    
    #    labelencoder_Y=LabelEncoder()
    #    y=labelencoder_Y.fit_transform(y)
    #    
        #Splittinvg thedataset into train_test_split
        from sklearn.cross_validation import train_test_split
        X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=.2,random_state=0)
    #    
    #    #Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc_x= StandardScaler()
        X_train=sc_x.fit_transform(X_train)
        X_test=sc_x.transform(X_test)
        
        
        
    except Exception as e:
        print(e)

def model(X_train,y_train):
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import mean_squared_error
        classifier=RandomForestClassifier(n_estimators=10,random_state=0)
        classifier.fit(X_train,y_train)
        
    #    #Create Decision tree model
    #    classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
    #    classifier.fit(X_train,y_train)
    #    y_pred=classifier.predict(X_test)
    #    from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
    
    
         import pickle
         # save the model to disk
         filename = 'finalized_model.sav'
         pickle.dump(classifier, open(filename, 'wb'))        
         y_pred = regressor.predict(X_test)
       
        #Mean Squared error
         print("Mean squared error: %.2f"
          % mean_squared_error(y_test, y_pred))
         
    except Exception as e:
        print(e)
        
def Predict(filename,X_test,y_test):
    # load the model from disk
    classifier = pickle.load(open(filename, 'rb'))
    
    y_pred=classifier.predict(X_test)
    
    result = classifier.score(X_test, Y_test)
    print(result)   
    
    #Accuracy and confusion matrix
    cm= confusion_matrix(y_test,y_pred)
    acc=accuracy_score(y_test,y_pred)
    class_report= classification_report(y_test,y_pred)
