# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 19:23:24 2021

@author: eranb
"""
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import math


#Assuming file.csv is valid and full with valid values
class NaiveBasyesOperator:
    
    def __init__(self,csv_root):    
        self.Tester = self.NaiveBayesProvider(csv_root)
        self.Tester.build_database()
        self.Tester.build_categories_probabilities()
    
    def evaluate_accuracy(self):
        y_test_prediction = np.apply_along_axis(self.Tester.make_prediction, 1, self.Tester.X_test)
        number_of_corrects_predictions = len(y_test_prediction[y_test_prediction==self.Tester.y_test])
        size_of_test = len(self.Tester.y_test)
        return (number_of_corrects_predictions/float(size_of_test))*100
    
    
    
        
    class NaiveBayesProvider:
      def __init__(self,csv_root):
          self.dataset = pd.read_csv(csv_root,',')
          X = self.dataset.iloc[:,:-1].values
          y = self.dataset.iloc[:,-1].values
               
          self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y, test_size=0.3, random_state=6)
          
          self.all_categories = np.unique(self.y_train)
          self.database3d_array = np.zeros((len(self.all_categories), len(self.X_train[0]), 2))
          self.categories_probabilities = np.zeros(len(self.all_categories))
          self.ctgry_to_index = {}
          self.index_to_ctgry = {}
          
          
          for i,category in enumerate(self.all_categories):
              self.ctgry_to_index[category] = i
              self.index_to_ctgry[i] = category
          
          
      def build_database(self):##check the iteration and the values of the 3d array(category_index)
          for category_index,category in enumerate(self.all_categories):
              X_category = self.X_train[self.y_train==category]
              for feature_index,feature in enumerate(X_category.T): #iterate over columns
                  
                  feature_mean = self.mean(feature)
                  feature_std = self.std(feature,feature_mean)
                  
                  self.database3d_array[category_index][feature_index][0]= feature_mean
                  self.database3d_array[category_index][feature_index][1]= feature_std
                  
      def build_categories_probabilities(self):
          for category in self.all_categories:
              len_of_category = len(self.y_train[self.y_train==category])
              len_of_labels = len(self.y_train)
              self.categories_probabilities[self.ctgry_to_index[category]] = (len_of_category*1.)/len_of_labels      
          
    
    
      # WHEN DEALING WITH AN INSTANCE:
            
      def predictator(self,category_index,instance):
          result = 1
          #print(self.database3d_array[category_index])
          for instance_index,arr_1d in enumerate(self.database3d_array[category_index]):
              feature_mean = arr_1d[0]
              feature_std = arr_1d[1]
              prob = self.calculate_probability(instance[instance_index],feature_mean,feature_std)
              result = result * prob    
          return result * self.probability(category_index) 
      
      def make_prediction(self,instance):
          max_prob = -1
          category_of_max = -1
          for category in self.all_categories:
              prediction = self.predictator(self.ctgry_to_index[category],instance)
              if max_prob < prediction :
                  max_prob = prediction
                  category_of_max = category
          return category_of_max    
              
      def mean(self,feature):
          m = len(feature)
          sum = 0
          for num in feature:
            sum+=num
          return sum*1./m   
      def std(self,feature,feature_mean):
          variance = sum([(x-feature_mean)**2 for x in feature]) / float(len(feature)-1)
          return math.sqrt(variance)
          
         
      def calculate_probability(self,x, mean, stdev):
         exponent = math.exp(-((x-mean)**2 / (2 * stdev**2 )))
         return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent
      
      def probability(self,category_index):
         return self.categories_probabilities[category_index]
          
     

    
Operator = NaiveBasyesOperator(csv_root="diabetes.csv")  
accuracy = Operator.evaluate_accuracy()
print(accuracy) 



#Operator2 = NaiveBasyesOperator(csv_root = dataset)
#accuracy = Operator2.evaluate_accuracy()
#print(accuracy)