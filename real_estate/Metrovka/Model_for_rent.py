import numpy as np
import pandas as pd
from transliterate import translit
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
from sklearn import datasets, linear_model,metrics
from sklearn.cross_validation import train_test_split
from sklearn.utils import check_arrays
from math import ceil

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = check_arrays(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

f = '/home/nastja/Dataroot/Price_calculator/Metrovka/places_kiev.json'
data = pd.read_json(path_or_buf = f)
data.drop(['aget_id','bedrooms_count','country','description','house_number','id','images','kind','kitchen_square','live_square','offer_type','rayon','region','slug','status','status_by_owner','street', 'subdistrict','user_data', 'user_id'], inplace=True, axis=1)
data = data.loc[data['type'] == 1]

headers = data.columns.values
for i in headers: data[i] = data[i].apply(lambda x: 0 if x == 'None' else x)


#categorical data into numbers

#material
#dummies = pd.get_dummies(data['material']).rename(columns=lambda x: 'material_' + str(x))
#data = pd.concat([data, dummies], axis=1)
#data = data.drop(['material','material_0'],axis=1)
data = data.drop(['material'],axis=1)

#city (only Kiev)	
data['city'] = data['city'].apply(lambda x: translit(x,"ru", reversed=True))
data = data.loc[data['city'] =='Kiev']
data = data.drop(['city'],axis=1)

#district
data['district'] = data['district'].apply(lambda x: translit(x,"ru", reversed=True))
data['district'] = data['district'].astype(np.str).replace(to_replace=' rajon', value='', regex=True) #remove ' rajon'
data['district'] = data['district'].astype(np.str).replace(to_replace='\'', value='', regex=True) #remove '
dummies = pd.get_dummies(data['district']).rename(columns=lambda x: 'district_' + str(x))
data = pd.concat([data, dummies], axis=1)
data = data.drop(['district','district_','district_nan'],axis=1)

#without_fee: convert True/False to 0/1
data['without_fee'] = data['without_fee'].apply(lambda x: 1 if x == 'True' else 0)




#cleaning data

#price to UAN
currency_to_UAH= {'UAH':1, 'USD':24.7000, 'EUR':27.0000}
data['currency'] = data['currency'].apply(lambda x: currency_to_UAH[x])
data['price'] = data.apply(lambda row: (row['price']*row['currency']),axis=1)
data = data.drop(['currency'],axis=1)

#values to numeric
data = data.convert_objects(convert_numeric=True)

#fill flats with 0 square by mean 
#dictionary: {number of rooms: mean square}
square_mean_5000 = dict(data.loc[data['price'] <= 5000].groupby(['rooms'])['square'].mean()) 
del square_mean_5000[0]
data['square'] = data.apply(lambda row: square_mean_5000[int(row['rooms'])] if (row['square'] == 0 and row['price'] <= 5000 and int(row['rooms']) in square_mean_5000.keys()) else int(row['square']),axis=1)

square_mean_10000 = dict(data.loc[data['price'] > 5000].loc[data['price'] <= 10000].groupby(['rooms'])['square'].mean()) 
data['square'] = data.apply(lambda row: square_mean_10000[int(row['rooms'])] if (row['square'] == 0 and row['price'] <= 10000 and int(row['rooms']) in square_mean_10000.keys()) else int(row['square']),axis=1)

square_mean_20000 = dict(data.loc[data['price'] > 10000].loc[data['price'] <= 15000].groupby(['rooms'])['square'].mean()) 
data['square'] = data.apply(lambda row: square_mean_20000[int(row['rooms'])] if (row['square'] == 0 and row['price'] <= 15000 and int(row['rooms']) in square_mean_20000.keys()) else int(row['square']),axis=1)

square_mean_40000 = dict(data.loc[data['price'] > 15000].loc[data['price'] <= 40000].groupby(['rooms'])['square'].mean()) 
del square_mean_40000[0]
data['square'] = data.apply(lambda row: square_mean_40000[int(row['rooms'])] if (row['square'] == 0 and row['price'] <= 40000 and int(row['rooms']) in square_mean_40000.keys()) else int(row['square']),axis=1)

square_mean_60000 = dict(data.loc[data['price'] > 40000].loc[data['price'] <= 60000].groupby(['rooms'])['square'].mean()) 
data['square'] = data.apply(lambda row: square_mean_60000[int(row['rooms'])] if (row['square'] == 0 and row['price'] <= 60000 and int(row['rooms']) in square_mean_60000.keys()) else int(row['square']),axis=1)

square_mean_80000 = dict(data.loc[data['price'] > 60000].loc[data['price'] <= 80000].groupby(['rooms'])['square'].mean()) 
data['square'] = data.apply(lambda row: square_mean_80000[int(row['rooms'])] if (row['square'] == 0 and row['price'] <= 80000 and int(row['rooms']) in square_mean_80000.keys()) else int(row['square']),axis=1)

square_mean_100000 = dict(data.loc[data['price'] > 80000].groupby(['rooms'])['square'].mean()) 
data['square'] = data.apply(lambda row: square_mean_100000[int(row['rooms'])] if (row['square'] == 0 and int(row['rooms']) in square_mean_100000.keys()) else int(row['square']),axis=1)

square_mean = dict(data.groupby(['rooms'])['square'].mean()) 
del square_mean[0]
data['square'] = data.apply(lambda row: square_mean[int(row['rooms'])] if (row['square'] == 0 and int(row['rooms']) in square_mean.keys()) else int(row['square']),axis=1)

#fill flats with 0 rooms by mean 
data['rooms'] = data.apply(lambda row: row['rooms'] if row['rooms'] else min(square_mean.items(), key=lambda (_, v): abs(v - row['square']))[0],axis=1)

#remove outliers
data = data.loc[data['price'] < 100000]
data = data.loc[data['price'] > 1500]
data['price_per_room'] = data.apply(lambda row: row['price']/float(row['rooms']),axis=1)
data['square_per_room'] = data.apply(lambda row: row['square']*float(row['rooms']),axis=1)
data = data.loc[data['price_per_room'] < 30000]
data = data.loc[data['square_per_room'] < 1500]

#remove not important features
data = data.drop(['without_fee'],axis=1) 
data = data.drop(['price_per_room'],axis=1)
data = data.drop(['square_per_room'],axis=1)

#get prices
data_price = data['price'] 	
data = data.drop(['price','type'], axis=1)

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

