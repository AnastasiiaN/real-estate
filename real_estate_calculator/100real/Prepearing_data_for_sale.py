import numpy as np
import pandas as pd
from transliterate import translit
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
from sklearn import datasets, linear_model,metrics
import json

file_name = '/home/nastja/Dataroot/Price_calculator/flat_for_sale.json'

with open(file_name, 'rb') as f: data = f.readlines()

data = map(lambda x: x.rstrip().split(']['), data)
data =  [item for sublist in data for item in sublist]
data = map(lambda x: x.rstrip(','), data)
data = ','.join(data)
data = pd.read_json(data)


#TV to boolean
data['TV'] = data['TV'].apply(lambda x: translit(x[0],"ru", reversed=True) if x else '')
data['TV'] = data['TV'].apply(lambda x: 1 if x == 'da' else 0)

#get district
#if len(adress)==2 - street, city
#if len(adress)==3 - street, numb. of house / district, city
#if len(adress)==4,5,6 - n-1-district

data['adress'] = data['adress'].apply(lambda x: translit(x[0],"ru", reversed=True).split(',') if x else [])
data['district'] = data['adress'].apply(lambda x: x[-2] if (len(x)>2 and not(x[1][1].isdigit())) else 0)
data['district'] = data['district'].astype(np.str).replace(to_replace=' \(tsentr\)', value='', regex=True) #remove '  (tsentr)'
data['district'] = data['district'].apply(lambda x: ' Podol\'skij' if x== ' Podol' else x) #Podol==Podolskij
data = data.drop(['adress'],axis=1)

#bldType
data['bldType'] = data['bldType'].apply(lambda x: translit(x[0],"ru", reversed=True) if x else '')

#floor
data['floor'] = data['floor'].apply(lambda x: translit(x[0],"ru", reversed=True).split('/') if x else '')
data['floor_count'] = data['floor'].apply(lambda x: x[1] if len(x)==2 else 0)
data['floor'] = data['floor'].apply(lambda x: x[0] if len(x)!=0 else 0)
data['mezzanine'] = data.apply(lambda row: 1 if row['floor'] == 'bel\'etazh' else 0,axis=1)
data['floor'] = data['floor'].apply(lambda x: 0 if (x == 'tsokol\'' or x =='bel\'etazh' or x == 'polupodval' or x == 'podval' ) else x) #tsokol'== 0, bel'etazh == 0

#furniture to boolean
data['furniture'] = data['furniture'].apply(lambda x: translit(x[0],"ru", reversed=True) if x else '')
data['furniture'] = data['furniture'].apply(lambda x: 1 if x == 'Est\'' else 0)

#parking to boolean
data['parking'] = data['parking'].apply(lambda x: translit(x[0],"ru", reversed=True) if x else '')
data['parking'] = data['parking'].apply(lambda x: 1 if x == 'da' else 0)

#phone to boolean
data['phone'] = data['phone'].apply(lambda x: translit(x[0],"ru", reversed=True) if x else '')
data['phone'] = data['phone'].apply(lambda x: 1 if x == 'da' else 0)

#refrigerator to boolean
data['refrigerator'] = data['refrigerator'].apply(lambda x: translit(x[0],"ru", reversed=True) if x else '')
data['refrigerator'] = data['refrigerator'].apply(lambda x: 1 if x == 'da' else 0)

#rooms
data['rooms'] = data['rooms'].apply(lambda x: translit(x[0],"ru", reversed=True).split('/')[0] if x else 0)

#square
data['square'] = data['square'].apply(lambda x: x[0].split('/') if x else 0)
data['total_square'] = data['square'].apply(lambda x: x[0] if x else 0)
data['living_square'] = data['square'].apply(lambda x: x[1] if (x and len(x) == 3) else 0)
data['kitchen_square'] = data['square'].apply(lambda x: x[2] if (x and len(x)==3) else x[1] if (x and len(x)==2) else 0)
data = data.drop(['square'],axis=1)

#price
data['price'] = data['price'].apply(lambda x: translit(x,"ru", reversed=True).replace(' ','')).astype(np.str).replace(to_replace=r'([\*]*grn\.)', value='', regex=True)

#materials_list, materials_values
data['materials_list'] = data['materials_list'].apply(lambda x: translit(','.join(x),"ru", reversed=True).split(','))
data['materials_values'] = data['materials_values'].apply(lambda x: translit(','.join(x),"ru", reversed=True).split(','))
data['materials_'] = data.apply(lambda row: zip(row['materials_list'],row['materials_values']),axis=1)
data['materials_'] = data['materials_'].apply(lambda x: dict(x))
data['wall_material'] = data['materials_'].apply(lambda x: x['Material sten:'] if ('Material sten:' in x.keys()) else 0)
data['floor_material'] = data['materials_'].apply(lambda x: x['Material pola:'] if ('Material pola:' in x.keys()) else 0)
data['overlapping_material'] = data['materials_'].apply(lambda x: x['Material perekrytij:'] if ('Material perekrytij:' in x.keys()) else 0)
data = data.drop(['materials_list','materials_values','materials_'],axis=1)

#state_list, state_values
data['state_list'] = data['state_list'].apply(lambda x: translit(','.join(x),"ru", reversed=True).split(','))
data['state_values'] = data['state_values'].apply(lambda x: translit(','.join(x),"ru", reversed=True).split(','))
data['state_'] = data.apply(lambda row: zip(row['state_list'],row['state_values']),axis=1)
data['state_'] = data['state_'].apply(lambda x: dict(x))
data['level_count'] = data['state_'].apply(lambda x: x['Kolichestvo urovnej:'] if ('Kolichestvo urovnej:' in x.keys()) else 0)
data['state'] = data['state_'].apply(lambda x: x['Remont (sostojanie):'] if ('Remont (sostojanie):' in x.keys()) else 0)
data = data.drop(['state_list','state_values','state_'],axis=1)

#wc_list, wc_values
data['wc_list'] = data['wc_list'].apply(lambda x: translit(','.join(x),"ru", reversed=True).split(','))
data['wc_values'] = data['wc_values'].apply(lambda x: translit(','.join(x),"ru", reversed=True).split(','))
data['wc_'] = data.apply(lambda row: zip(row['wc_list'],row['wc_values']),axis=1)
data['wc_'] = data['wc_'].apply(lambda x: dict(x))
#data['count'] = data['wc_'].apply(lambda x: x['Kolichestvo'] if ('Kolichestvo' in x.keys()) else 0)
data['wc_count'] = data['wc_'].apply(lambda x: x['Kolichestvo sanuzlov:'] if ('Kolichestvo sanuzlov:' in x.keys()) else 0)
data['wc_types'] = data['wc_'].apply(lambda x: x[' tip sanuzlov:'] if (' tip sanuzlov:' in x.keys()) else 0)
data['wc_type'] = data['wc_'].apply(lambda x: x['Tip sanuzla:'] if ('Tip sanuzla:' in x.keys()) else 0)
data['balcony'] = data['wc_'].apply(lambda x: 1 if ('Balkon:' in x.keys()) else 0)
data['wc_type'] = data.apply(lambda row: row['wc_types'].lstrip(' ') if row['wc_types'] else row['wc_type'] if row['wc_type'] else 0,axis=1)
data = data.drop(['wc_list','wc_values','wc_','wc_types'],axis=1)

#metro
data['metro'] = data['metro'].apply(lambda x: translit(','.join(x),"ru", reversed=True).split(','))



import json	
dlist = data.to_dict('records')
dlist = [json.dumps(record)+"\n" for record in dlist]
open('100real_for_sale_ready.json','w').writelines(dlist)

