import numpy as np
import pandas as pd
from transliterate import translit
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
from sklearn.cross_validation import train_test_split
from sklearn import datasets, linear_model,metrics
from sklearn.utils import check_arrays

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = check_arrays(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

file_name = '100real_for_rent_ready.json'

with open(file_name, 'rb') as f: data = f.readlines()
data = map(lambda x: x.rstrip(), data)
data = '['+','.join(data)+']'
data = pd.read_json(data)


#categorical data into numbers

#district

dummies = pd.get_dummies(data['district']).rename(columns=lambda x: 'district_' + str(x))
data = pd.concat([data, dummies], axis=1)
data = data.drop(['district','district_0'],axis=1)

#bldType
dummies = pd.get_dummies(data['bldType']).rename(columns=lambda x: 'bldType_' + str(x))
data = pd.concat([data, dummies], axis=1)
data = data.drop(['bldType','bldType_'],axis=1)

#floor_material
#dummies = pd.get_dummies(data['floor_material']).rename(columns=lambda x: 'floor_material_' + str(x))
#data = pd.concat([data, dummies], axis=1)
#data = data.drop(['floor_material','floor_material_0'],axis=1)
data = data.drop(['floor_material'],axis=1)

#state
dummies = pd.get_dummies(data['state']).rename(columns=lambda x: 'state_' + str(x))
data = pd.concat([data, dummies], axis=1)
data = data.drop(['state','state_0'],axis=1)

#wall_material
#dummies = pd.get_dummies(data['wall_material']).rename(columns=lambda x: 'wall_material_' + str(x))
#data = pd.concat([data, dummies], axis=1)
#data = data.drop(['wall_material','wall_material_0'],axis=1)
data = data.drop(['wall_material'], axis=1)


#wc_type
#dummies = pd.get_dummies(data['wc_type']).rename(columns=lambda x: 'wc_type_' + str(x))
#data = pd.concat([data, dummies], axis=1)
#data = data.drop(['wc_type','wc_type_0'],axis=1)
data = data.drop(['wc_type'],axis=1)


#overlapping_material
#dummies = pd.get_dummies(data['overlapping_material']).rename(columns=lambda x: 'overlapping_material_' + str(x))
#data = pd.concat([data, dummies], axis=1)
#data = data.drop(['overlapping_material','overlapping_material_0'],axis=1)
data = data.drop(['overlapping_material'],axis=1)

#metro
data['metro'] = data['metro'].apply(lambda x: [i.lstrip(' ') for i in x])
metro =  [item for sublist in data['metro'] for item in sublist]
metro = sorted(list(set(metro)))
for i in metro: data[i] = data['metro'].apply(lambda x: 1 if x[0]==i else 0)
data = data.drop(['metro',''],axis=1)


#cleaning data

#dictionary: {number of rooms: mean square}
square_mean = dict(data.groupby(['rooms'])['total_square'].mean()) 

#fill flats with 0 square by mean 
data['total_square'] = data.apply(lambda row: square_mean[int(row['rooms'])] if (row['total_square'] == 0 and int(row['rooms']) in square_mean.keys()) else int(row['total_square']),axis=1)

#fill flats with 0 rooms by mean 
data['rooms'] = data.apply(lambda row: row['rooms']if row['rooms'] else min(square_mean.items(), key=lambda (_, v): abs(v - row['total_square']))[0],axis=1)

#fill flats with 0 wc by 1
data['wc_count'] = data['wc_count'].apply(lambda x: x if x else 1)

#remove outliers
data = data.loc[data['rooms'] < 11]
data = data.loc[data['price'] < 250000]
data = data.loc[data['price'] > 1000]
data['price_per_room'] = data.apply(lambda row: row['price']/float(row['rooms']),axis=1)
data = data.loc[data['price_per_room'] < 50000]
data['square_per_room'] = data.apply(lambda row: row['total_square']*float(row['rooms']),axis=1)
data = data.loc[data['square_per_room'] < 2000]

#remove not important features
data = data.drop(['price_per_room'], axis=1)
data = data.drop(['living_square'], axis=1)
data = data.drop(['kitchen_square'], axis=1)
data = data.drop(['balcony'], axis=1)
data = data.drop(['mezzanine'], axis=1)
data = data.drop(['refrigerator'], axis=1)
data = data.drop(['wc_count'], axis=1)



#get prices
data_price = data['price'] 	
data = data.drop(['price'], axis=1)


#model: statsmodel
re = sm.OLS(data_price, data).fit()
print re.summary()

#model: sklearn
regr = linear_model.LinearRegression(normalize=True)
regr.fit(data,data_price)
print ('\nMean square error: %.2f' % np.mean((regr.predict(data) - data_price.as_matrix())**2))
print ('Variance score: %.2f' % regr.score(data, data_price))
print ('Mean absolute percentage error: %.2f' % mean_absolute_percentage_error(data_price, regr.predict(data)))

#split data to training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, data_price, test_size=0.33, random_state=42)
regr = linear_model.LinearRegression(normalize=True)
regr.fit(X_train,y_train)
print '\n\nSplitting data to training and test sets:'
print ('Mean square error: %.2f' % np.mean((regr.predict(X_test) - y_test)**2))
print ('Variance score: %.2f' % regr.score(X_test, y_test))
print ('Mean absolute percentage error: %.2f' % mean_absolute_percentage_error(y_test, regr.predict(X_test)))

