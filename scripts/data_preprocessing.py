import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv('/home/vmac/lab3/dataset/dataset.csv',index_col=0)   

DF = df.drop_duplicates()
DF = DF.reset_index(drop=True)

cat_columns = []
num_columns = []

for column_name in DF.columns:
    if (DF[column_name].dtypes == object):
        cat_columns +=[column_name]
    else:
        num_columns +=[column_name]

#Применим OneHotEncoder к категориальным признакам
qwer = pd.get_dummies(DF[cat_columns])
DF= DF.join(qwer)

DF.drop(columns=['product_code', 'attribute_0', 'attribute_1'], inplace=True)

X, y = DF.drop(columns = ['failure']).values,DF['failure'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


with open('/home/vmac/lab3/dataset/X_train.npy', 'wb') as f:    
    np.save(f, X_train)
with open('/home/vmac/lab3/dataset/X_test.npy', 'wb') as f:    
    np.save(f, X_test)
with open('/home/vmac/lab3/dataset/y_train.npy', 'wb') as f:    
    np.save(f, y_train)
with open('/home/vmac/lab3/dataset/y_test.npy', 'wb') as f:    
    np.save(f, y_test)