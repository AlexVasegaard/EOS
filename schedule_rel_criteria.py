def schedule_criteria(number_of_requests, total_days, number_of_requests_0, satellite_swath):
    #create random request areas - with corresponding data
    import pandas as pd
    import random
    from random import uniform, choices, randint, shuffle
    from scipy.stats import poisson
    import numpy as np
    
    random.seed(42)
    np.random.seed(42)
    
    #total number of requests 
    total_request = number_of_requests + number_of_requests_0
    
    #request ID
    from ID_create import ID_create
    req_ID = ID_create(total_request)
    
    #location data    #from https://latitudelongitude.org/
    #ukraine
    req_loc = list()
    divisions=8
    total_request_div = int(total_request/divisions)
    data_location_which = 4 #-2 = updated article, -1 = reviewved article, 0=article, 1 = ukraine airbus, 3 = world normal, 4 = world concentrated
    
     
    
    
    if data_location_which == 4: 
        import sys
        data_dir = sys.path[0][0:-7] + "data/worldloc.xlsx"
        loc_data = pd.read_excel(data_dir)  #from_world_gen_data.py
        req_loc1 = loc_data.sample(total_request)
        for i in range(total_request):
            req_loc.append([req_loc1['lat'].iloc[i], req_loc1['lng'].iloc[i]]) 
            
    
    if data_location_which == -2:  #updated updated article 
        divisions = 7
        total_request_div = int(total_request/divisions)
        for a in range(0,7): #-2 due to 3 part last uniform
            if a == 0:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(43,44), uniform(1,2)])   #Toulouse
            if a ==1:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(48.5,49.5), uniform(1.5,2.5)])   #paris
            if a == 2:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(43.2,43.8), uniform(7,8)])   #nice
            if a == 3:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(41.59,51.03), uniform(-4.65,9.45)])   #entire france
            if a == 4:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(56.7,57.3), uniform(9,10)])   #aalborg 57.048, 9.9187
            if a == 5:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(55,56), uniform(12,13)])      #københavn 55.67594, 12.56553
            #2part
            if a == 6: 
                left = total_request - (6)*int(total_request/divisions)
                for i in range(0, left):
                    req_loc.append([uniform(48.2,49), uniform(-4,-3.5)]) #rennes

    
    if data_location_which == -1:  #updated article 
        divisions = 7
        total_request_div = int(total_request/divisions)
        for a in range(0,divisions-2): #-2 due to 3 part last uniform
            if a == 0:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(56.7,57.3), uniform(9.5,10)])   #aalborg 57.048, 9.9187
            if a == 1:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(54,58), uniform(8,15)])    #entire denmark 54.76906 to 57.72093 and longitude from 8.24402 to 14.70664
            if a == 2:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(43.2,43.8), uniform(1.2,1.8)])   #Toulouse 43.60426, 1.44367
            #2part
            if a == 3: 
                left = total_request - (3)*int(total_request/divisions)
                for i in range(0, left):
                    req_loc.append([uniform(41.59,51.03), uniform(-4.65,9.45)]) #entire france Latitude from 41.59101 to 51.03457 and longitude from -4.65 to 9.45

    
    
    if data_location_which == 0:
        for a in range(0,divisions-1): #-1 due to 2 part uniform
            if a == 0:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(56.5,57.5), uniform(9,10)])   #aalborg 57.048, 9.9187
            if a == 1:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(55,56), uniform(12,13)])      #københavn 55.67594, 12.56553
            if a == 2:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(54.769,57.72), uniform(8.24,14.70)])    #entire denmark 54.76906 to 57.72093 and longitude from 8.24402 to 14.70664
            if a == 3:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(43,44), uniform(1,2)])   #Toulouse 43.60426, 1.44367
            if a == 4:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(48,49.5), uniform(1.5,3)]) #Paris 48.85341, 2.3488
            if a == 5:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(43,44), uniform(7,8)])   #Nice 43.70313, 7.26608
            #2part
            if a == 6: 
                left = total_request - (divisions-2)*int(total_request/divisions)
                for i in range(0, left):
                    req_loc.append([uniform(41.59,51.03), uniform(-4.65,9.45)]) #entire france Latitude from 41.59101 to 51.03457 and longitude from -4.65 to 9.45
    
    if data_location_which == 1:
        for a in range(0,divisions-1): #-1 due to 2 part uniform
            if a == 0:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(44.4,52.2), uniform(22.2,40.2)])   #ukraine
            if a == 1:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(35.9,42), uniform(25.9,44.5)])      #turkey
            if a == 2:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(43.6,48.2), uniform(20.5,28.8)])    #romania
            if a == 3:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(41.4,44), uniform(22.7,28.3)])   #bulgaria
            if a == 4:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(51.8,55.9), uniform(23.7,32)]) #belarus
            if a == 5:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(54.5,59.4), uniform(21.2,27.7)])   #lituania--estonia
            #2part
            if a == 6: 
                left = total_request - (divisions-2)*int(total_request/divisions)
                for i in range(0, left):
                    req_loc.append([uniform(44.5,59), uniform(30.3, 37.6)]) #eastern part of eastern europe 
        
        #world
    if data_location_which == 3:
        req_loc = list()
        divisions=8
        total_request_div = int(total_request/divisions)
        for a in range(0,divisions-1): #-1 due to 2 part uniform
            if a == 0:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(35,45), uniform(-85, -70)])   #New York
            if a == 1:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(40, 60), uniform(0,20)])      #Denmark
            if a == 2:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(30,50), uniform(-10, 10)])    #Toulouse
            if a == 3:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(30,45), uniform(135, 145)])   #Tokyo
            if a == 4:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(-45,-35), uniform(140, 150)]) #Melbourne
            if a == 5:
                for i in range(0, total_request_div):
                    req_loc.append([uniform(30, 40), uniform(-130, -110)])   #Amazonas rainforrest
            #2part
            if a == 6: 
                left = total_request - (divisions-2)*int(total_request/divisions)
                for i in range(0, left):
                    req_loc.append([uniform(-90,90), uniform(-180, 180)]) #uniform 
    #len(req_loc)
    shuffle(req_loc)
    
    #day
    if number_of_requests != 0:
        time_stamp = list([0]*number_of_requests_0)
        divisor = list([randint(1,number_of_requests) for i in range(0, total_days-1)])
        divisor.append(0)
        divisor.append(number_of_requests)
        divisor.sort()
        k=0
        for l in range(0,len(divisor)-1):
            k=k+1
            for k in range(divisor[l],divisor[l+1]):
                time_stamp.append(k)
        len(time_stamp)
    else:
        time_stamp = list([0]*number_of_requests_0)
        
    #stereo request? 0 not a stereo req, 1 it is a stereo req
    req_stereo = poisson.rvs(mu=0.1, size=total_request)  #mu = 0.1
    
    #area of request
    req_area = np.zeros(total_request)
    for i in range(total_request):
        req_area[i] = uniform(1,1000)
    req_area = list(req_area)
# =============================================================================
#     divisions=10
#     total_request_div = int(total_request/divisions)
#     for a in range(0,divisions-1): #-1 due to 2 part uniform
#         if a == 0 or a == 1 or a == 2 or a == 3 or a == 4 or a == 5 or a == 6:
#             for i in range(0, total_request_div):
#                 req_area.append(randint(1,400)) #km2
#         if a == 7:
#             for i in range(0, total_request_div):
#                 req_area.append(randint(900,2500)) #km2
#         #2part
#         if a == 8: 
#             left = total_request - (divisions-2)*int(total_request/divisions)
#             for i in range(0, left):
#                 req_area.append(randint(400,900)) #km2
#     shuffle(req_area)
# =============================================================================
    
            
    #difficulty of acquirering request - shape difficulty/#strips
    req_strips = list()
    for i in range(total_request):
        pct_strip = 2.25*np.sqrt(req_area[i])/np.sqrt(satellite_swath) #2.25 is magical number.. it increases the possibility of strips acquisitions when area distribution is small
        req_strips.append(int(np.ceil(pct_strip)))
    strips = np.array(req_strips)
    strips[req_stereo>0] = 1  
    req_strips = strips          #if it is a stereo request it is  
                                 #not acquired in multiple strips
    
    #duration of acquirering the data
    ##calculated with duration function: dur = diff*acquisition speed*sqrt(area)
    #units = 2*5.1sec/m*sqrt(5000 m^2)
    #where acquisition speed is assumed to be 
# =============================================================================
#     acq_speed = 0.0700
#     
#     req_duration = np.zeros(total_request)
#     for i in range(total_request):
#         req_duration[i] = acq_speed * np.sqrt(req_area[i])
#     req_duration = list(req_duration)
#     
# =============================================================================
    req_duration = np.zeros(total_request)
    for i in range(total_request):
        req_duration[i] = uniform(2,8)
    req_duration = list(req_duration)
    
    #original priority fitting to airbus distribution of eastern europe
    if (data_location_which in [-2, -1, 0, 3, 4]): #article
        req_pri = list()
        population = [1, 2, 3, 4]               #population = [1, 2, 3, 4, 5, 6, 7]
        weights = [0.25, 0.25, 0.25, 0.25]  #weights = [0.13, 0.18, 0.5, 0.1, 0.206, 0.22, 0.157]
        #weights = [0.023, 0.018, 0.005, 0.1, 0.646, 0.152, 0.057]
        for i in range(total_request):
            req_pri.append(choices(population, weights)[0])
        
        #modified priority and customertype fitting to electre framework
        req_mod_pri = list() 
        req_mod_type = list()
        for i in range(total_request):
            if (req_pri[i] == 1):
                req_mod_type.append(1)
                req_mod_pri.append(1)
            if (req_pri[i] == 2):
                req_mod_type.append(1)
                req_mod_pri.append(2)
            if (req_pri[i] == 3):
                req_mod_type.append(2)
                req_mod_pri.append(3)
            if (req_pri[i] == 4):
                req_mod_type.append(2)
                req_mod_pri.append(4)

    if data_location_which == 1 : #airbus
        req_pri = list()
        population = [1, 2, 3, 4, 5, 6, 7]
        #weights = [0.13, 0.18, 0.5, 0.1, 0.206, 0.22, 0.157]
        weights = [0.013, 0.018, 0.005, 0.1, 0.646, 0.152, 0.057]
        for i in range(total_request):
            req_pri.append(choices(population, weights)[0])
        
        #modified priority and customertype fitting to electre framework
        req_mod_pri = list() 
        req_mod_type = list()
        for i in range(total_request):
            if (req_pri[i] == 1):
                req_mod_type.append(1)
                req_mod_pri.append(1)
            if (req_pri[i] == 2):
                req_mod_type.append(2)
                req_mod_pri.append(2)
            if (req_pri[i] == 3):
                req_mod_type.append(3)
                req_mod_pri.append(3)
            if (req_pri[i] == 4):
                req_mod_type.append(3)
                req_mod_pri.append(4)
            if (req_pri[i] == 5):
                req_mod_type.append(3)
                req_mod_pri.append(5)
            if (req_pri[i] == 6):
                req_mod_type.append(4)
                req_mod_pri.append(6)
            if (req_pri[i] == 7):
                req_mod_type.append(4)
                req_mod_pri.append(7)

    #price
    req_price = list()
    for i in range(total_request):
        req_price.append(randint(500,1500))
    #zero profit for priority 1 
    #for j in list(np.where(np.array(req_mod_pri)==1)[0]):
    #    req_price[j] = 0  #100000000
    #if data_location_which == 1: #airbus and T2
    #    for j in list(np.where(np.array(req_mod_pri)==2)[0]):
    #        req_price[j] = 0  #1000000
    
    
    df = pd.DataFrame({ "ID"                    : req_ID ,
                       "acquired"               : list(map(int, np.zeros(total_request))),
                       "reachable"              : list(map(int, np.zeros(total_request))),
                       "request location"       : req_loc,
                       "day"                    : time_stamp,
                       "area"                   : req_area,
                       "stereo"                 : req_stereo,
                       "strips"                 : req_strips,
                       "duration"               : req_duration,
                       "priority"               : req_pri,
                       "priority mod"           : req_mod_pri,
                       "customer type mod"      : req_mod_type,
                       "price"                  : req_price})
    return(df)