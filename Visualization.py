# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 20:03:33 2021

@author: renji
"""


import matplotlib.pyplot as plt
import pandas as pd

data_csv = pd.read_csv('D:\\TCE\\sem 6\\Data Science using Python\\Predict price\\train.csv')
print(data_csv)

"null values finding"
df_nulls = pd.read_csv('D:\\TCE\\sem 6\\Data Science using Python\\Predict price\\train.csv',)
df_nulls.isnull().sum()


"histogram"
import matplotlib.pyplot as plt
import pandas as pd
mobile_data=pd.read_csv('D:\\TCE\\sem 6\\Data Science using Python\\Predict price\\train.csv')

plt.hist(mobile_data['ram'], color='green', edgecolor='white',bins=5)
plt.title('Histogram')
plt.xlabel('RAM')
plt.ylabel('Frequency')
plt.show()


"scatter"
import matplotlib.pyplot as plt

mobile_data=pd.read_csv('D:\\TCE\\sem 6\\Data Science using Python\\Predict price\\train.csv')

plt.scatter(mobile_data['ram'],mobile_data['price_range'],c='red')
plt.title('Scatter Plot')
plt.xlabel('RAM')
plt.ylabel('Price')
plt.show()

"bar chart"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mobile_data=pd.read_csv('D:\\TCE\\sem 6\\Data Science using Python\\Predict price\\train.csv')

x = mobile_data['ram'] 
y = mobile_data['price_range']
plt.bar(x,y)
plt.show()

"piechart"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mobile_data=pd.read_csv('D:\\TCE\\sem 6\\Data Science using Python\\Predict price\\train.csv')
y = mobile_data['price_range']
plt.pie(y)
plt.show()


"seaborn.pointplot"
import seaborn as sns
mobile_data=pd.read_csv('D:\\TCE\\sem 6\\Data Science using Python\\Predict price\\train.csv')
sns.pointplot(y="ram", x="price_range", data=mobile_data)


"seaborn.barplot"
import seaborn as sns
mobile_data=pd.read_csv('D:\\TCE\\sem 6\\Data Science using Python\\Predict price\\train.csv')
plt.figure(figsize=(15,6));
sns.barplot( x= "n_cores", y = "battery_power" ,hue="price_range", data=mobile_data)
plt.xticks(rotation=90);




