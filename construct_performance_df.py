#construction of performance data frame 
def construct_performance_df(df, seconds_gran, location_slots, time_slots, distance, height_satellite, 
                             hours_ahead, weather, generate_weather):
    max_cloud_cover = 50
    
    #df=DF_i
    import ephem
    import pandas as pd
    import numpy as np
    import random
    import datetime
    
    random.seed(42)
    np.random.seed(42)
    
    #Setup lists for panda df
    ID = list()
    stereo = list()
    sat = list()
    loc_from = list()
    loc_to = list()
    time = list()
    area_req = list()
    strips_req = list()
    dur_req = list()
    dist_req = list()
    angle_req = list()
    sun_elevation = list()
    pri_req = list()
    pri_mod_req = list()
    type_mod_req = list()
    price_req = list() 
    wait_req = list()
    
    uncertainty_req = list()
    cloud_req_real = list()
    cloud_req = list()
    
        
    for k in range(0, len(location_slots)):
        for n in range(0,len(location_slots[0])):
            #w_n = random.randint(1,100)
            for i in range(0,df.shape[0]):
                #checking whether i attempt is feasible by starting and closest end attempt.
                #nxt_attempt = int(np.ceil(df.iloc[i]["duration"]/seconds_gran))
                if (~np.isnan(distance[n,i,k])): #((df.shape[0] > i + nxt_attempt) and ~np.isnan(distance[n,i,k]) and ~np.isnan(distance[n,i+nxt_attempt,k])): #attempt distance feasible (1st point)
                    ID.append(df.iloc[i]["ID"])

                    ### Satellite relative data
                    sat.append(k)
                    loc_from.append(location_slots[k][n])
                    loc_to.append(df.iloc[i]["request location"])
                    time.append(time_slots[n])
                    area_req.append(df.iloc[i]["area"])
                    strips_req.append(df.iloc[i]["strips"])
                    dur_req.append(df.iloc[i]["duration"])  #should also be dependent on angle that is n
                    dist_req.append(distance[n,i,k])
                    angle_i = np.degrees(np.arcsin(distance[n,i,k]/np.sqrt(height_satellite**2 + distance[n,i,k]**2)))
                    angle_req.append(angle_i)
                    
                    obs = ephem.Observer()
                    obs.lat = str(df.iloc[i]["request location"][0])
                    obs.long = str(df.iloc[i]["request location"][1])
                    obs.date = time_slots[n]
                    sun = ephem.Sun(obs)
                    sun.compute(obs)
                    sun_angle = np.degrees(sun.alt) # Convert Radians to degrees
                    sun_elevation.append(sun_angle)
                    
                    uncertainty_req.append(n/len(location_slots[0])) #represented by the number of timeslot
                    
                
                    
                    cloud_req.append(0)
                    cloud_req_real.append(0)
                
                    ### constant regular data
                    stereo.append(df.iloc[i]["stereo"])
                    pri_req.append(df.iloc[i]["priority"])
                    pri_mod_req.append(df.iloc[i]["priority mod"])
                    type_mod_req.append(df.iloc[i]["customer type mod"])
                    price_req.append(df.iloc[i]["price"]) 
                    wait_req.append(df.iloc[i]["waiting time"])
        
    performance_df = pd.DataFrame({"ID"                     : ID,
                                   "stereo"                 : stereo,
                                   "satellite"              : sat,
                                   "satellite location"     : loc_from,
                                   "request location"       : loc_to,
                                   "time"                   : time,
                                   "area"                   : area_req,
                                   "strips"                 : strips_req,
                                   "duration"               : dur_req,
                                   "distance"               : dist_req,
                                   "angle"                  : angle_req,
                                   "sun elevation"          : sun_elevation,
                                   "cloud cover estimate"   : cloud_req,
                                   #      "wind"                   : wind_req,
                                   #      "humidity"               : humidity_req,
                                   #      "pressure"               : pressure_req,
                                   #      "temperature"            : temperature_req,
                                   "priority"               : pri_req,
                                   "priority mod"           : pri_mod_req,
                                   "customer type mod"      : type_mod_req,
                                   "price"                  : price_req,
                                   "waiting time"           : wait_req,
                                   "uncertainty"            : uncertainty_req,
                                   "cloud cover real"       : cloud_req_real })
        
    ##remove bad sun elevation attempts
    performance_df = performance_df.drop(performance_df[performance_df["sun elevation"] < 0].index)
    performance_df = performance_df.reset_index(drop =True)
    
    def cloud_gen(lat, long, parameter, alpha = 3, beta =4.5):
        lat1 = np.cos(((lat+90)/180)*20) #to mimic higher around equator and one upper and lower quantile
        long1 = np.cos(((long+180)/360)*50)
        lat2 = np.cos(((lat+90)/180)*50)
        long2 = np.cos(((long+180)/360)*30)
        lat3 = np.cos(((lat+90)/180)*100)
        long3 = np.cos(((long+180)/360)*80)

        cloud = (((3*lat1 *long1 + 1.5*lat1 + 2*lat2*long2 + 1*lat3*long3)*100/8 + 70)*2 -150)/1.5 # it now has a range from -50 to 150
        #cloud = 110-(((lat1 * long1 + lat2*long2 + 2)  / (alpha)) -(beta-alpha)/beta)*110 #to mimic scewedness! 
        #cloud_stoc2=cloud
        cloud_stoc1 = cloud + parameter*random.randint(-10,10) #to generate bias in both ends
        cloud_stoc2 = max(min(cloud_stoc1 + random.randint(-5,5),100),0)
        return(cloud_stoc2)
        
    if (weather == False and generate_weather == True):
        cloud_req_real = list()
        cloud_req = list()
        unique_locs = np.unique(performance_df["request location"])
        unique_locs_list = np.concatenate(unique_locs).reshape((len(unique_locs),2))
        unique_weather = list()
        similarity_parameter = np.random.uniform(0,1)
        for i in range(0,len(unique_locs)):
            cloud_loc = cloud_gen(unique_locs[0][0], unique_locs[0][1], similarity_parameter) 
            unique_weather.append(cloud_loc)
        for i in range(0,len(performance_df)):
            which_unique_loc = np.where(np.sum(unique_locs_list == performance_df["request location"][i], axis = 1) == 2)[0][0]
            w_i = max(min(unique_weather[which_unique_loc] + random.randint(-5,5), 100),0)
            error_time = random.randint(-10,10)*performance_df['uncertainty'][i]
            w_i_error = max(min(w_i+error_time, 100),0)
            cloud_req.append(w_i)
            cloud_req_real.append(w_i_error)
        performance_df['cloud cover estimate'] = cloud_req
        performance_df['cloud cover real'] = cloud_req_real
        

    #insert weather forecast for the feasible locations - note the forecast 
    #obtained closest to the acquisition time will be set for that alternative
    #weather data ASSUMED
    if weather == True:
        #forecast or historical?
        weather_type = "historic" #"forecast"
        if weather_type == "forecast":
            import requests
            #API_key = ["c34e8aff26eef394ae38ac01b7ddccd6","69309ebdc6bc38636ed3f7e046189ce1",
            #           "7e7e3f7c0ae7daded084d4a8bfdd4a14","8dd7579adbe1dd18b024d96ab802201a",
            #           'a5d2345f547f66370e770ef14b5fea5c'] 
            unique_locs = np.unique(performance_df["request location"])
            forecast_t_v = [[] for _ in range(2)] 
            keys = 0
            for i in range(0,len(unique_locs)):
                if (i>50 and i<100):
                    keys = 1
                if (i>100 and i<150):
                    keys = 2
                if (i>150 and i<200):
                    keys = 3
                if (i>200 and i<250):
                    keys = 4
                if (i>250 and i<300):
                    keys = 5
                
                call_str = "http://api.openweathermap.org/data/2.5/forecast?lat="+str(unique_locs[i][0])+"&lon="+str(unique_locs[i][1])+"&appid="+API_key[keys]
                rep = requests.get(call_str)
                json_dict = rep.json()
                number_of_forecasts = int(np.ceil(hours_ahead/3))
                for f in range(number_of_forecasts):
                    forecasts = json_dict["list"][f]
                    forecast_t_v[0].append(forecasts["dt_txt"])
                    forecast_t_v[1].append(forecasts['clouds']['all'])
                    
            unique_locs_list = np.concatenate(unique_locs).reshape((len(unique_locs),2))
            times = np.array(list(set(forecast_t_v[0])), dtype = "datetime64[ns]")
            for i in range(0,len(performance_df)):
                which_f = np.argmin(times - np.array(performance_df["time"][i], dtype ="datetime64[ns]"))
                which_unique_loc = np.where(np.sum(unique_locs_list == performance_df["request location"][i], axis = 1) == 2)[0][0]
                performance_df["cloud cover estimate"][i] = forecast_t_v[1][(which_unique_loc*number_of_forecasts)+which_f]
                
            #we assume the observed weather is somewhat near the forecast
            for i in range(0,len(performance_df)):
                performance_df["cloud cover real"][i] = max(min(performance_df["cloud cover estimate"][i]+(n/len(location_slots[0]))*random.randint(-20,20), 100),0)
              
                
                
                
                
                
        if weather_type == "historic":
            import requests
            #API_key = "b88d9a00178f4b63bf976b78eb70c103"
            unique_locs = np.unique(performance_df["request location"])
            unique_weather = list()
            #for each unique requests - collect data
            t_from = int(str(time_slots[0])[11:13])
            t_to = t_from + int(np.ceil(hours_ahead))+1
            for i in range(0,len(unique_locs)):
                call_str = "https://api.weatherbit.io/v2.0/history/hourly?lat="+str(unique_locs[i][0])+"&lon="+str(unique_locs[i][1])+"&start_date="+str(time_slots[0])[0:10]+"&end_date="+str(time_slots[0]+datetime.timedelta(days=1))[0:10]+"&tz=local&key=b88d9a00178f4b63bf976b78eb70c103"
                rep = requests.get(call_str)
                json_dict = rep.json()
                
                weather_list = list()
                for iii in range(t_from, t_to):
                    weather_list.append(json_dict["data"][iii]['clouds']) 
                unique_weather.append(weather_list)
                print(i, " out of ", len(unique_locs), ":", weather_list)
            #for all requests - allocatae information
            unique_locs_list = np.concatenate(unique_locs).reshape((len(unique_locs),2))
            for i in range(0,len(performance_df)):
                which_h = np.where(np.arange(t_from,t_to)==int(str(performance_df["time"][i])[11:13]))[0][0]
                which_unique_loc = np.where(np.sum(unique_locs_list == performance_df["request location"][i], axis = 1) == 2)[0][0]
                
                performance_df["cloud cover estimate"][i] = unique_weather[which_unique_loc][which_h]
                
            #we assume the observed weather is somewhat near the forecast
            for i in range(0,len(performance_df)):
                performance_df["cloud cover real"][i] = max(min(performance_df["cloud cover estimate"][i]+(n/len(location_slots[0]))*random.randint(-20,20), 100),0)
                
    
    ## remove bad weather attempts
    performance_df = performance_df.drop(performance_df[performance_df["cloud cover estimate"] > max_cloud_cover].index)
    
    ###### TOPOLOGICAL SORTING!!
    #sort by acq satellite and time, and reset index 
    performance_df = performance_df.sort_values(by = ["satellite", "time"]).reset_index(drop = True)
    
    sat_unique = np.unique(performance_df['satellite'])
    first_obs_time = pd.DataFrame({'time' : [0]*len(sat_unique)})
    for i in range(len(sat_unique)):
        first_obs_time.iloc[i] = performance_df['time'].iloc[np.where(performance_df['satellite'] == sat_unique[i])[0][0]]
    new_sat_names = np.array(first_obs_time.sort_values(by = 'time').index)
    new_sat_names = new_sat_names[::-1]
    
    sat_old = performance_df['satellite']
    sat_new = list()
    for i in range(0,len(performance_df)):
        idx_sat = int(np.where(np.isin(sat_unique, sat_old[i]))[0])
        if sat_old[i] == 0:
            sat_new.append(new_sat_names[idx_sat])
        elif sat_old[i] == 1:
            sat_new.append(new_sat_names[idx_sat])
        elif sat_old[i] == 2:
            sat_new.append(new_sat_names[idx_sat])
        elif sat_old[i] == 3:
            sat_new.append(new_sat_names[idx_sat])
            
    performance_df['satellite']=sat_new
    
    #sort by NEW acq satellite and time, and reset index 
    performance_df = performance_df.sort_values(by = ["satellite", "time"]).reset_index()
    
    
    
    
    return(performance_df)
