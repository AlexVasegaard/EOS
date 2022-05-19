#create world location data set to draw random locations from
import pandas as pd
from random import uniform, choices, randint, shuffle
from scipy.stats import poisson
import numpy as np

min_limit_city = 15000

#location data generation and cleaning
loc_data = pd.read_excel("//ID.AAU.DK/Users/QF82BM/Desktop/robust SIASP paper/worldcities.xlsx")  #from https://simplemaps.com/data/world-cities
#loc_data.iloc[np.where(loc_data.iloc[:,0] == 'Aalborg')[0][0],:]
#loc_data.info
range_loc = 1.5
l1 = np.where(loc_data['lat']+range_loc>90)[0]
l2 = np.where(loc_data['lng']+range_loc>180)[0]
l3 = np.where(loc_data['lat']-range_loc<-90)[0]
l4 = np.where(loc_data['lng']-range_loc<-180)[0]
if len(l1) != 0:
    loc_data.iloc[l1,2] = np.array(len(l1)*[range_loc])
if len(l2) != 0:
    loc_data.iloc[l2,2] = np.array(len(l2)*[range_loc])
if len(l3) != 0:
    loc_data.iloc[l3,2] = np.array(len(l3)*[range_loc])
if len(l4) != 0:
    loc_data.iloc[l4,2] = np.array(len(l4)*[range_loc])


size = loc_data['population']>min_limit_city


city = list()
country = list()
lat = list()
lng = list()
for i in range(0, np.sum(size)):
    size_i = loc_data['population'][size].iloc[i] 
    if size_i <= 100000:
        adds = 2
        for i in range(adds):
            city.append(loc_data['city_ascii'][size].iloc[i])
            country.append(loc_data['country'][size].iloc[i])
            lat.append(uniform(loc_data['lat'][size].iloc[i]-(adds/20),loc_data['lat'][size].iloc[i]+(adds/20)))
            lng.append(uniform(loc_data['lng'][size].iloc[i]-(adds/20),loc_data['lng'][size].iloc[i]+(adds/20)))
            
        
    elif size_i > 100000 and size_i <= 500000:
        adds = 5
        c_i  = loc_data['city_ascii'][size].iloc[i]
        cc_i = loc_data['country'][size].iloc[i]
        la_i = loc_data['lat'][size].iloc[i]
        lo_i = loc_data['lng'][size].iloc[i]
        for i in range(adds):
            city.append(c_i)
            country.append(cc_i)
            lat.append(uniform(la_i-(adds/20),la_i+(adds/20)))
            lng.append(uniform(lo_i-(adds/20),lo_i+(adds/20)))
         
    elif size_i > 500000 and size_i <= 1000000:
        adds = 10 
        c_i  = loc_data['city_ascii'][size].iloc[i]
        cc_i = loc_data['country'][size].iloc[i]
        la_i = loc_data['lat'][size].iloc[i]
        lo_i = loc_data['lng'][size].iloc[i]
        for i in range(adds):
            city.append(c_i)
            country.append(cc_i)
            lat.append(uniform(la_i-(adds/20),la_i+(adds/20)))
            lng.append(uniform(lo_i-(adds/20),lo_i+(adds/20)))
        
    elif size_i > 1000000 and size_i <= 5000000:
        adds = 15 
        c_i  = loc_data['city_ascii'][size].iloc[i]
        cc_i = loc_data['country'][size].iloc[i]
        la_i = loc_data['lat'][size].iloc[i]
        lo_i = loc_data['lng'][size].iloc[i]
        for i in range(adds):
            city.append(c_i)
            country.append(cc_i)
            lat.append(uniform(la_i-(adds/20),la_i+(adds/20)))
            lng.append(uniform(lo_i-(adds/20),lo_i+(adds/20)))
        
    elif size_i > 5000000 :
        adds = 30
        c_i  = loc_data['city_ascii'][size].iloc[i]
        cc_i = loc_data['country'][size].iloc[i]
        la_i = loc_data['lat'][size].iloc[i]
        lo_i = loc_data['lng'][size].iloc[i]
        for i in range(adds):
            city.append(c_i)
            country.append(cc_i)
            lat.append(uniform(la_i-(adds/20),la_i+(adds/20)))
            lng.append(uniform(lo_i-(adds/20),lo_i+(adds/20)))
        
        
loc_df = pd.DataFrame(data = {
        'city'     :   city,
        'country'  :   country,
        'lat'      :   lat,
        'lng'      :   lng
        }
)

loc_df.to_excel("//ID.AAU.DK/Users/QF82BM/Desktop/SPECIALE/data/worldloc.xlsx")

#import matplotlib.pyplot as plt
#plt.scatter(loc_df['lng'], loc_df['lat'])




