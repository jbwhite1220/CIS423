import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

from sklearn.metrics import f1_score#, balanced_accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier







#This class maps values in a column, numeric or categorical.
class MappingTransformer(BaseEstimator, TransformerMixin):
  
  def __init__(self, mapping_column, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?
    
    #now check to see if all keys are contained in column
    column_set = set(X[self.mapping_column])
    keys_not_found = set(self.mapping_dict.keys()) - column_set
    if keys_not_found:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain these keys as values {keys_not_found}\n")

    #now check to see if some keys are absent
    keys_absent = column_set -  set(self.mapping_dict.keys())
    if keys_absent:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values {keys_absent}\n")

    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
  
  
  
class OHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=False):  
    self.target_column = target_column
    self.dummy_na=dummy_na
    self.drop_first=drop_first
    pass
 
  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    dummy = X.copy()
    assert self.target_column in X.columns, f'{self.__class__.__name__}. Input Dataframe has no column named {self.target_column}'
    new = pd.get_dummies(dummy,
                               prefix=self.target_column,    #your choice
                               prefix_sep='_',     #your choice
                               columns=[self.target_column],
                               dummy_na=False,    #will try to impute later so leave NaNs in place
                               drop_first=False    #really should be True but could screw us up later
                               )
    
    return new

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def fit_transform(self, X, y = None):
      result = self.transform(X)
      return result
    
    
    
    
    
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_list, action='drop'):
    assert action in ['keep', 'drop'], f'{self.__class__.__name__} action {action} not in ["keep", "drop"]'
    self.column_list = column_list
    self.action = action
    
    pass
 
  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    dummy = X.copy()
    copy_col = self.column_list
    if self.action == 'keep':
      missing_col = set(copy_col).difference(X.columns)
      assert len(missing_col) == 0, f'{self.__class__.__name__}: input dataframe does not contain columns: {missing_col}'
      for item in dummy.columns:
        if item not in self.column_list:
          del dummy[item]
    
    else:
      missing_col = set(copy_col).difference(X.columns)
      if len(missing_col) != 0:
        print(f"\nWarning: {self.__class__.__name__}.attempting to drop nonexistent columns: {missing_col}.\n")
        for item in missing_col:
          copy_col.remove(item)

      dummy.drop(columns = copy_col, axis=1, inplace=True)

      


    return dummy

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def fit_transform(self, X, y = None):
      result = self.transform(X)
      return result
    
    
class TukeyTransformer(BaseEstimator, TransformerMixin):

  def __init__(self, target_column, fence):
    self.column_name = target_column
    self.mode = fence

  def transform(self, X):
    clean = X.copy()
    fig, ax = plt.subplots(1,1, figsize=(3,9))
    clean.boxplot(self.column_name, vert=True, ax=ax, grid=True)
    q1 = clean[self.column_name].quantile(0.25)
    q3 = clean[self.column_name].quantile(0.75)

    iqr = q3-q1
    outer_low = q1-3*iqr
    outer_high = q3+3*iqr
    inner_low = q1-1.5*iqr
    inner_high = q3+1.5*iqr
    if self.mode == 'outer': #or self.mode == 'inner':
      clean[self.column_name] = clean[self.column_name].clip(lower = outer_low, upper=outer_high)
      
    else:
      clean[self.column_name] = clean[self.column_name].clip(lower = inner_low, upper=inner_high)
      
    return clean

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  


class Sigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_name):
    self.column_name = column_name

  def transform(self, X):
    X_ = X.copy()
    m = X_[self.column_name].mean()
    sigma = X_[self.column_name].std()
    mins, maxs = (m-3*sigma, m+3*sigma)
    X_[self.column_name] = X_[self.column_name].clip(lower=mins, upper=maxs)
    return X_

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

    
class MinMaxTransformer(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass  #takes no arguments

  def transform(self, X):
    X_copy = X.copy()
    col_list = X_copy.columns
    
    for item in col_list:
      mi = X_copy[item].min()
      mx = X_copy[item].max()
      denom = (mx-mi)
      X_copy[item] -= mi
      X_copy[item] /= denom

    return X_copy
      

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
    
    
class KNNTransformer(BaseEstimator, TransformerMixin):
  def __init__(self,n_neighbors=5, weights="uniform"):
    #your code
    self.imputer = KNNImputer(n_neighbors=n_neighbors,        #a rough guess
                     weights=weights,    #could alternatively have distance factor in
                     add_indicator=False)  #do not add extra column for NaN

    
  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    X_copy = X.copy()
    col_list = X_copy.columns
    imputed_data = self.imputer.fit_transform(X_copy) 
    new_data = pd.DataFrame(imputed_data,columns=col_list) 
    return new_data
    
    
  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
    
    

def find_random_state(features_df, labels, n=200):
  model = KNeighborsClassifier(n_neighbors=5)
  var = []  #collect test_error/train_error where error based on F1 score

  #2 minutes
  for i in range(1, 200):
      train_X, test_X, train_y, test_y = train_test_split(features_df, labels, test_size=0.2, shuffle=True,
                                                      random_state=i, stratify=labels)
      model.fit(train_X, train_y)  #train model
      train_pred = model.predict(train_X)           #predict against training set
      test_pred = model.predict(test_X)             #predict against test set
      train_f1 = f1_score(train_y, train_pred)   #F1 on training predictions
      test_f1 = f1_score(test_y, test_pred)      #F1 on test predictions
      f1_ratio = test_f1/train_f1          #take the ratio
      var.append(f1_ratio)

  rs_value = sum(var)/len(var)  #get average ratio value
  rs_value  #0.8666447504722037

  idx = np.array(abs(var - rs_value)).argmin()  #find the index of the smallest value
  return idx

titanic_transformer = Pipeline(steps=[
    ('drop', DropColumnsTransformer(['Age', 'Gender', 'Class', 'Joined', 'Married',  'Fare'], 'keep')),
    ('gender', MappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('class', MappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('ohe', OHETransformer(target_column='Joined')),
    ('age', TukeyTransformer(target_column='Age', fence='outer')), #from chapter 4
    ('fare', TukeyTransformer(target_column='Fare', fence='outer')), #from chapter 4
    ('minmax', MinMaxTransformer()),  #from chapter 5
    ('imputer', KNNTransformer())  #from chapter 6
    ], verbose=True)

customer_transformer = Pipeline(steps=[
    ('id', DropColumnsTransformer(column_list=['ID'])),  #you may need to add an action if you have no default
    ('os', OHETransformer(target_column='OS')),
    ('isp', OHETransformer(target_column='ISP')),
    ('level', MappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high':2})),
    ('gender', MappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('time spent', TukeyTransformer('Time Spent', 'inner')),
    ('minmax', MinMaxTransformer()),
    ('imputer', KNNTransformer())
    ], verbose=True)

def dataset_setup(full_table, label_column_name:str, the_transformer, rs, ts=.2):
  #your code below
  table_features = full_table.drop(columns=label_column_name)
  labels = full_table[label_column_name].to_list()
  X_train, X_test, y_train, y_test = train_test_split(table_features, labels, test_size=ts, shuffle=True,
                                                    random_state=rs, stratify=labels)
  
  X_train_transformed = the_transformer.fit_transform(X_train)
  X_test_transformed = the_transformer.fit_transform(X_test)

  X_trained_numpy = X_train_transformed.to_numpy()
  X_test_numpy = X_test_transformed.to_numpy()
  y_train_numpy = np.array(y_train)
  y_test_numpy = np.array(y_test)
  return X_trained_numpy, X_test_numpy, y_train_numpy, y_test_numpy

def titanic_setup(titanic_table, transformer=titanic_transformer, rs=40, ts=.2):
  
  return dataset_setup(titanic_table, 'Survived', titanic_transformer, rs, ts=ts)

def customer_setup(customer_table, transformer=customer_transformer, rs=76, ts=.2):
  
  return dataset_setup(customer_table, 'Rating', customer_transformer, rs, ts=ts)
