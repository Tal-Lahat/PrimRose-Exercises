# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:19:08 2020

@author: Lea
"""

# Part 1
import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
    
chipo = pd.read_csv(url, sep = '\t')
chipo.head(10)
# chipo['choice_description'][4]
chipo.info()#

# OR

chipo.shape[0]
# 4622 observations
chipo.shape[1]
chipo.columns
chipo.index
chipo.item_name.value_counts().head(1)
mostOrd = chipo.item_name.value_counts().max() #or mostOrd = chipo["item_name"].max()
mostOrd
chipo.choice_description.value_counts().head()
total_items_orders = chipo.quantity.sum()
total_items_orders

# dollarizer = lambda x: float(x[1:-1])
chipo.item_price = chipo.item_price.apply(lambda x: float(x[1:-1]))

(chipo.item_price*chipo.quantity).sum()
chipo.order_id.value_counts().count()

order_grouped = chipo.groupby(by=['order_id'])
order_grouped.mean()['item_price']

# Or 

#chipo.groupby(by=['order_id']).sum().mean()['item_price']
chipo.item_name.value_counts().count()


# Part 2
import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
    
chipo = pd.read_csv(url, sep = '\t')
# clean the item_price column and transform it in a float
prices = [float(value[1 : -1]) for value in chipo.item_price]

# reassign the column with the cleaned prices
chipo.item_price = prices 

# make the comparison
chipo10 = chipo[chipo['item_price'] > 10.00]
chipo10.head()

len(chipo10)
# delete the duplicates in item_name and quantity
chipo_filtered = chipo.drop_duplicates(['item_name','quantity'])

## select only the ones with quantity equals to 1
#price_per_item = chipo_filtered[chipo_filtered.quantity == 1]

#
price_per_item = chipo_filtered[['item_name', 'item_price']]

# sort the values from the most to less expensive
#price_per_item.sort_values(by = "item_price", ascending = False)
#chipo.item_name.sort_values()

# OR

chipo.sort_values(by = "item_name")
chipo.sort_values(by = "item_price", ascending = False).head(1)
chipo_salad = chipo[chipo.item_name == "Veggie Salad Bowl"]

len(chipo_salad)
chipo_drink_steak_bowl = chipo[(chipo.item_name == "Canned Soda") & (chipo.quantity > 1)]
len(chipo_drink_steak_bowl)

# Part 3
import pandas as pd
users = pd.read_table('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user', 
                      sep='|', index_col='user_id')
users.head()
users.groupby('occupation').age.mean()
# create a function
def gender_to_numeric(x):
    if x == 'M':
        return 1
    if x == 'F':
        return 0

# apply the function to the gender column and create a new column
users['gender_n'] = users['gender'].apply(gender_to_numeric)


a = users.groupby('occupation').gender_n.sum() / users.occupation.value_counts() * 100 

# sort to the most male 
a.sort_values(ascending = False)
users.groupby('occupation').age.agg(['min', 'max'])
users.groupby(['occupation', 'gender']).age.mean()
# create a data frame and apply count to gender
gender_ocup = users.groupby(['occupation', 'gender']).agg({'gender': 'count'})

# create a DataFrame and apply count for each occupation
occup_count = users.groupby(['occupation']).agg('count')

# divide the gender_ocup per the occup_count and multiply per 100
occup_gender = gender_ocup.div(occup_count, level = "occupation") * 100

# present all rows from the 'gender column'
occup_gender.loc[: , 'gender']


# Part 4
import pandas as pd
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
data1 = pd.DataFrame(raw_data_1, columns = ['subject_id', 'first_name', 'last_name'])
data2 = pd.DataFrame(raw_data_2, columns = ['subject_id', 'first_name', 'last_name'])
data3 = pd.DataFrame(raw_data_3, columns = ['subject_id','test_id'])

data3
all_data = pd.concat([data1, data2])
all_data
all_data_col = pd.concat([data1, data2], axis = 1)
all_data_col
data3
merged1 = pd.merge(all_data, data3, on='subject_id')
merged2 = pd.merge(data1, data2, on='subject_id', how='inner')
merged3 = pd.merge(data1, data2, on='subject_id', how='outer')


# part 5
import pandas as pd
import numpy as np
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = pd.read_csv(url)

iris.head()
# 1. sepal_length (in cm)
# 2. sepal_width (in cm)
# 3. petal_length (in cm)
# 4. petal_width (in cm)
# 5. class

iris.columns = ['sepal_length','sepal_width', 'petal_length', 'petal_width', 'class']
iris.head()
pd.isnull(iris).sum()
# nice no missing value
iris.iloc[10:30,2:3] = np.nan
iris.head(20)
iris.petal_length.fillna(1, inplace = True)
iris
del iris['class']
iris.head()
iris.iloc[0:3 ,:] = np.nan
iris.head()
iris = iris.dropna(how='any')
iris.head()
iris = iris.reset_index(drop = True)
iris.head()


