# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 13:30:34 2021

@author: Tal

Pandas H.W Part 1 - Basic """ """
"""
#%%
"""
1.Getting & knowing your data
"""
#S1:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
#S2,3,4: Use pd.read_csv(fpath, sep='\t') or pd.read_table(fpath)
# imp_data=pd.read_table("https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv")
imp_data=pd.read_csv("https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv", sep='\t')
chipo=pd.DataFrame(imp_data)   
print(chipo.head(10))
#%%
#S5,6: How many observations we got? Number of Columns?
print(chipo.describe)
# There are 4622 rows x 5 columns, so I think the answer is 4622 Observations? or maybe we use count:
print('Number of Observations:\n',max(chipo.count()))
print('Number of columns:\n',max(chipo.count(1)))
#S7. print the names of all the columns 
print('The columns names are:\n',chipo.columns)
#%%
#S8. How was the dataset indexed?
print('This how it was indexed:\n',chipo.index)
#S9. Which was the most ordered item?
print(chipo.describe(include='all')) 
chipo_des=chipo.describe(include='all')
# chipo_des.iloc[2, 2]
print("The most ordered item is:\n",chipo_des.loc['top','item_name'])
"""Chicken Bowl"""
#S10. How many were ordered?
# print("Chicken Bowl was ordered",chipo_des.loc['freq','item_name'],"times\n") """This was not correct"""               
print("Chicken Bowl was ordered",chipo[chipo.item_name=='Chicken Bowl'].quantity.sum(),"times\n")
#S11. What was the most ordered item in the choice_description column?
print("The most ordered item is:\n",chipo_des.loc['top','choice_description'])
#S12. How many items were orderd in total?
# count=0
# for i in chipo.quantity:
    # count=count+i  
print("Total",chipo.quantity.sum(),"Items were ordered\n")
#%%
#S13. Turn the item price into a float ("chipo.dtypes")
chipo_float = lambda x: float(x[1:-1]) #anonymous function, returns what is written after ":", we want to skip the $ sign
chipo.item_price = chipo.item_price.transform(chipo_float)
print(chipo.item_price.dtypes)
#%% Another way
chipo.item_price=chipo.item_price.str.slice(start=1).astype(float)
print(chipo['item_price'].dtype)
#%%
#S14. How much was the revenue for the period in the dataset?
print("Total revenue is "+str((chipo.quantity*chipo.item_price).sum())+"$\n")
#S15. How many orders were made in the period?
print("There were a total of "+str(chipo.order_id.max())+" orders\n")
#S16. What is the average amount per order?
# print(chipo.groupby('item_name').item_price.mean().mean())
chipo['revenue'] = chipo['quantity'] * chipo['item_price']
print('Average amount per order is:\n',chipo.groupby(by=['order_id']).sum().mean()['revenue'])
#S17. How many different items are sold?   
print(chipo.item_name.value_counts().count(),"Different items sold\n")
#%%
"""2. Filtering & Sorting """
#S1,2,3: Import and assign DF into chipo """Just ran S1-3 from Getting & knowing your data"""
import pandas as pd
imp_data=pd.read_csv("https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv", sep='\t')
chipo=pd.DataFrame(imp_data)   
#Step 4. How many products cost more than $10.00?
chipo.item_price=chipo.item_price.str.slice(start=1).astype(float)
#%% 
chipo['price_per_item']=chipo.item_price/chipo.quantity
product_prices=chipo.groupby('item_name').max()
print((product_prices.price_per_item >10).sum()) 
#%%
# Step 5. What is the price of each item?
#print a data frame with only two columns item_name and item_price
drop_duplicate=chipo.drop_duplicates(['item_name','price_per_item'])
chipo_ppi=drop_duplicate.loc[:,('item_name','price_per_item')]
print('Price per item:\n',chipo_ppi)
#%%
#Step 6. Sort by the name of the item
chipo_sort_name=chipo.item_name.sort_values()
print(chipo_sort_name)
#Step 7. What was the quantity of the most expensive item ordered?
print(chipo.sort_values(by = "item_price", ascending = False).head(1))
#or
chipo_sort_values=chipo.sort_values(by = "item_price", ascending = False) 
print(chipo_sort_values.quantity.iloc[0])
#Step 8. How many times were a Veggie Salad Bowl ordered?
print("Veggie Salad Bowl ordered was ordered",chipo[chipo.item_name=='Veggie Salad Bowl'].quantity.sum(),"times\n")
#Step 9. How many times people orderd more than one Canned Soda?    
Soda=(chipo[chipo.item_name=='Canned Soda'].quantity >1).sum()
print(Soda)
#Or 
#chipo_drink_steak_bowl = chipo[(chipo.item_name == "Canned Soda") & (chipo.quantity > 1)]
#%%
"""3. Grouping"""
#Step 1. Import the necessary libraries
#Step 2. Import the dataset from this https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user
#Step 3. Assign it to a variable called users.
import pandas as pd
imp_data=pd.read_csv("https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user", sep='|')
users=pd.DataFrame(imp_data)   
#Step 4. Discover what is the mean age per occupation
print('The mean age per occupation is:\n'+str(users.groupby('occupation').age.agg('mean')))
#Step 5. Discover the Male ratio per occupation and sort it from the most to the least
male_users=users[users['gender']=='M']
x=male_users.groupby(['occupation'])['gender'].count()
y=users.groupby(['occupation'])['gender'].count()
result=((x/y)*100).round(2).sort_values(ascending=False)
print(result)
#or
x=users.groupby(['occupation','gender'])['gender'].count()
y=users.groupby(['occupation'])['gender'].count()
r=((x/y)*100).round(2).sort_values(ascending=False)
print(r[:2])
#Step 6. For each occupation, calculate the minimum and maximum ages
print(users.groupby('occupation').age.agg(['min', 'max']))
#Step 7. For each combination of occupation and gender, calculate the mean age
print(users.groupby(['occupation','gender']).age.mean())
#Step 8. For each occupation present the percentage of women and men
# create a data frame and apply count to gender
gender_ocup = users.groupby(['occupation', 'gender']).agg({'gender': 'count'})

# create a DataFrame and apply count for each occupation
occup_count = users.groupby(['occupation']).agg('count')

# divide the gender_ocup per the occup_count and multiply per 100
occup_gender = gender_ocup.div(occup_count, level = "occupation") * 100

# present all rows from the 'gender column'
print(occup_gender.loc[: , 'gender'])
#%%
# x=users.groupby(['occupation','gender'])['gender'].count()
# y=users.groupby(['occupation'])['gender'].count()
# r=((x/y)*100).round(2).sort_values(ascending=False)
# print(r.drop(columns=['gender']))
#%%
""""4. Merge"""
# Step 1. Import the necessary libraries
import numpy as np
import pandas as pd
# Step 2. Create the 3 DataFrames based on the followin raw data
raw_data_1 = {
'subject_id': ['1', '2', '3', '4', '5'],
'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}
raw_data_2 = {
'subject_id': ['4', '5', '6', '7', '8'],
'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}
raw_data_3 = {
'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}
# Step 3. Assign each to a variable called data1, data2, data3
data1=pd.DataFrame(raw_data_1)
data2=pd.DataFrame(raw_data_2)
data3=pd.DataFrame(raw_data_3)
# Step 4. Join the two dataframes along rows and assign all_data חיבור על ציר איקס
all_data = pd.concat([data1, data2])
# Step 5. Join the two dataframes along columns and assing to all_data_col חיבור על ציר וואי
all_data_col=pd.concat([data1,data2],axis=1)
# Step 6. Print data3
print(data3)
# Step 7. Merge all_data and data3 along the subject_id value  
"""With merge you can pick the column with similar data to be merged"""
merged=pd.merge(all_data,data3,how='inner',on='subject_id')
# Step 8. Merge only the data that has the same 'subject_id' on both data1 and data2
merged2=pd.merge(data1,data2,how='inner',on='subject_id')
# Step 9. Merge all values in data1 and data2, with matching records from both sides where available.
merged3=pd.merge(data1, data2, on='subject_id', how='outer')
#%%
"""5. Deleting"""
#Step 1. Import the necessary libraries
#Step 2. Import the dataset from this
#https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
#Step 3. Assign it to a variable called iris
import numpy as np
import pandas as pd
iris=pd.read_csv(r'C:\Users\leora\Desktop\טל\קורס MACHINE LEARNING\פרימרוז\שבוע 4\iris.data',sep=",")
#or
#%%
iris=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
#Step 4. Create columns for the dataset
#1. sepal_length (in cm)
#. sepal_width (in cm)
#3. petal_length (in cm)
#4. petal_width (in cm)
#5. class
iris.columns=['sepal_length','sepal_width','petal_length','petal_width','class']
#Step 5. Is there any missing value in the dataframe? irish.ifno()
print(pd.isnull(iris).sum())
#%%
#Step 6. Lets set the values of the rows 10 to 29 of the column 'petal_length' to NaN
iris.iloc[10:29,2]=pd.np.NAN
#%%
#Step 7. Good, now lets substitute the NaN values to 1.0
iris.fillna(1, inplace = True)
#Step 8. Now let's delete the column class
del iris['class']
#%%
#Step 9. Set the first 3 rows as NaN
iris.iloc[0:3 ,:] = np.nan
#or iris.iloc[:3] = np.nan
print(iris.head())
#%%
#Step 10. Delete the rows that have NaN
iris.dropna(inplace=True)
#Step 11. Reset the index so it begins with 0 again
iris = iris.reset_index(drop = True) #The Drop=True removes old index
