#overall functions

def scenario(seconds_gran = 20, number_of_requests_0 = 1000, NORAD_ids = [38755, 40053], weather_real = False, simplify = True, 
                   schedule_start = [2021,7,21,9,40], 
                   hours_horizon = 8, 
                   max_off_nadir_angle = 30, #degrees
                   height_satellite = 694,   #km
                   rotation_speed = 30/12, #degrees per second #per https://directory.eoportal.org/web/eoportal/satellite-missions/s/spot-6-7
                   cam_resolution = 1, #m^2 per pixel
                   capacity_limit = 1000000, #in mega byte
                   satellite_swath = 3600,
                   map_generation=True): #compute maps?
    # NORAD_ids = [38012, 39019]
    # seconds_gran =10
    # number_of_requests_0 =30
    # weather_real = False
    ## entire setup ##
    import datetime
    #from satellite_tle import fetch_tle_from_celestrak
    import requests
    
    
    
    TLEs = list()
    for i in range(0, len(NORAD_ids)):
        #tle = requests.get('https://www.celestrak.com/satcat/tle.php?CATNR={}'.format(NORAD_ids[i]), #old version
        tle = requests.get('https://www.celestrak.com/NORAD/elements/gp.php?CATNR={}&FORMAT=TLE'.format(NORAD_ids[i]), #new version
                           verify=True, 
                           timeout=20)
        TLE = tle.text.split('\r\n')
        TLEs.append(TLE)
    #NORAD_ids = [38012, 39019, 38755, 40053]
    
    ################### SCHEDULE INPUT ##################
    #starting time for horizon
    import random
    random.seed(42)
    
    #time_stochastic = random.uniform(0,24)
    start_schedule = datetime.datetime(schedule_start[0],schedule_start[1],schedule_start[2],schedule_start[3],schedule_start[4]) #- datetime.timedelta(hours=int(time_stochastic))
    #start_schedule = datetime.datetime(2020,5,8,12,50)
    #datetime.datetime.utcnow() - datetime.timedelta(days=total_days+day) - datetime.timedelta(hours = 18)
    #hours ahead to schedule 
    hours_ahead = hours_horizon
    print(start_schedule, 'to', start_schedule + datetime.timedelta(hours=int(hours_ahead)))
    #Note, Dates always use Universal Time, NOT local time
    #granularity of scheduling (discretization),
    #i.e. time segmentation
    #number of satellite
    number_of_satellites = len(NORAD_ids)
    #compute maps?
    map_generation=True
    ####################################################
    
    ################## SATELLITE INPUT #################
    #TLE of satellite 1
    # =============================================================================
    #     name_1 = "pleiades 1A";
    #     line1_1 = "1 38012U 11076F   19303.06642174 -.00000056  00000-0 -23405-5 0  9997";
    #     line2_1 = "2 38012  98.1883  15.9149 0001494  90.8918 269.2452 14.58560404418937"
    #     
    #     #TLE of satellite 2
    #     name_2 = "pleiades 1B";
    #     line1_2 = "1 39019U 12068A   19303.10054518 -.00000046  00000-0 -91078-7 0  9993";
    #     line2_2 = "2 39019  98.1870  15.9140 0001220  88.0086 272.1258 14.58558050367792"
    #     
    #     #TLE of satellite 3
    #     name_3 = "SPOT 6";
    #     line1_3 = "1 38755U 12047A   19303.08272605 -.00000060  00000-0 -30989-5 0  9990";
    #     line2_3 = "2 38755  98.1980   8.4889 0000909  81.1977 278.9329 14.58563161380015"
    #     
    #     #TLE of satellite 4
    #     name_4 = "SPOT 7";
    #     line1_4 = "1 40053U 14034A   19303.11654248  .00001178  00000-0  26330-3 0  9995";
    #     line2_4 = "2 40053  98.2135   8.3331 0001054  68.4868 291.6410 14.58559748283956"
    # =============================================================================
    
    #SATELLITE SPECS assumed to be the same
    #no energy constraint - ease computation 
    #satellite cameras reachability
    #max_off_nadir_angle = 30 #degrees
    #height_satellite = 694   #km
    #rotation_speed = 30/12 #degrees per second #per https://directory.eoportal.org/web/eoportal/satellite-missions/s/spot-6-7
    #cam_resolution = 1 #m^2 per pixel
    #capacity_limit = 1000000 #in mega byte
    #satellite_swath = 3600
    ########################################################
    #allowed_weather = 50
    #succesful_weather = 10
    
    ############# DATA GENERATION INPUT ####################
    #number of requests from customers through # days!
    number_of_requests = 0
    #request already in database
    #number_of_requests_0 = 0
    ########################################################
    
    #packages
    import ephem
    from math import degrees, floor
    import folium
    import pandas as pd
    import numpy as np
    np.random.seed(42)
     
    #import timeit
    #data_time_start = timeit.default_timer() 
    
    ################### DATA GENERATION ####################
    #### schedule relative data ###
    from EOSpython.schedule_rel_criteria import schedule_criteria
    day=1
    total_days = 1
    df = schedule_criteria(number_of_requests, total_days, number_of_requests_0, satellite_swath)
    #df.info()
    avg_pri_df = [[list(np.mean(df[df["priority"] == j], axis = 0))[i] for i in [3,4,5,6,7,8,11]] for j in list(range(1,8))]
    #np_avg_pri_df = np.array(avg_pri_df)
    col_names_avg = list(df.columns[[4,5,6,7,8,9,12]])
    df_avg_pri = pd.DataFrame(
            {"measures": col_names_avg,
             "pri 1": avg_pri_df[0],
             "pri 2": avg_pri_df[1],
             "pri 3": avg_pri_df[2],
             "pri 4": avg_pri_df[3],
             "pri 5": avg_pri_df[4],
             "pri 6": avg_pri_df[5],
             "pri 7": avg_pri_df[6],
             }   
    )
    df_avg_pri.iloc[:,:4] 
    df_avg_pri.iloc[:,4:] 
    
    
    #### prelimenary data analysis begins - COLLECT VALID DATA
    #changing waiting time accordingly
    DF_i = df[(df["day"] <= day) & (df["acquired"] == 0)]
    pd.options.mode.chained_assignment = None
    
    added_wait = list()
    for i in range(0,DF_i.shape[0]):
        added_wait.append(random.randint(1,14))  #age distribution randomly uniform 1 to 14!!!!!!
    DF_i["waiting time"] = day - DF_i["day"] + np.array(added_wait)
    #stereo requests into multiple requests
    #DF_i["stereo"]>1
    
    
    #data_time_end_generation = data_time_start - timeit.default_timer() 
    
    
    ##Start map generation
    if map_generation == True:
        m = folium.Map(location=[20, 0], zoom_start=2) #tiles= 'Cartodb Positron'
        from EOSpython.plot_requests import plot_requests, plot_requests2
        plot_requests(m, df = DF_i, name = "request location", radius = 5)  
        #file is called all_requests.html
    
    
    
    
    
    ### satellite path computation
    
    sat_names = list()
    tle_rec = list()
    for i in range(0, len(TLEs)):
        sat_names.append(TLEs[i][0])
        tle_rec.append(ephem.readtle(TLEs[i][0], TLEs[i][1], TLEs[i][2]))
        
    increment = datetime.timedelta(seconds=seconds_gran)
    number_of_acq_points = floor((hours_ahead*60*60)/seconds_gran) #convert hours to seconds
    time_slots = list()
    location_slots = list()
    for i in range(0, len(NORAD_ids)):
        location_slots.append(list())
    for i in range(0,number_of_acq_points):
        time = start_schedule + (i)*increment
        time_slots.append(time)    
        for k in range(0, number_of_satellites):
            tle_rec[k].compute(time)
            location_slots[k].append([degrees(tle_rec[k].sublat), degrees(tle_rec[k].sublong)])    
    
    ### DATA COMPUTATION (satellite relative) 
    from EOSpython.distance_matrix import distance_matrix
    distance = distance_matrix(location_slots, DF_i, max_off_nadir_angle, height_satellite, number_of_satellites)
    print("number of reachable attempts:", np.sum(~np.isnan(distance)))
    
    #data_time_end_path = data_time_start - timeit.default_timer() 
    
    if weather_real == True:
        weather = True
        generate_weather = False
    else:
        weather = False
        generate_weather = True
        
    from EOSpython.construct_performance_df import construct_performance_df
    performance_df = construct_performance_df(DF_i, seconds_gran, location_slots, time_slots, 
                                              distance, height_satellite, hours_ahead, 
                                              weather = weather, generate_weather = generate_weather)
    print("number of attempts within thresholds:", performance_df.shape[0])
    print("number of requests", len(np.unique(performance_df['ID'])))
    #plot satellite path
    if map_generation == True:
        sat = folium.FeatureGroup(name = "Satellite path")
        colors = list(["royalblue", "black", "darkolivegreen", "green"])
        for k in range(0,len(location_slots)):
            for i in range(0,len(location_slots[k])):
                popup_text = "Satellite: {}<br> time: {}<br> possible acquisitions: <br>{}"
                popup_text = popup_text.format(sat_names[k],
                                               time_slots[i].strftime('%H:%M:%S-%m/%d/%Y'),
                                               list(np.where(performance_df["time"] == time_slots[i])[0]))
                sat.add_child(folium.CircleMarker(location=location_slots[k][i], radius = 4, opacity = 1-(i/(1.3*number_of_acq_points)),popup=folium.Popup(popup_text), color = colors[k]))
        m.add_child(sat)
        #m.add_child(folium.LayerControl())
        m.save("sat_path.html")
    
    if map_generation == True:
        plot_requests2(m, df = performance_df, name = "request location")  
        #file is called all_requests.html
    
    #data_time_end_preprocessing = data_time_start - timeit.default_timer() 
    
    
    #performance_df.info()
    ###################################################################
    ########################### LPP SETUP #############################
    ###################################################################
    
    from EOSpython.LPP_data_setup import LPP_data_multi
    LPP = LPP_data_multi(performance_df, number_of_acq_points, time_slots, location_slots, 
                 height_satellite, rotation_speed, seconds_gran,
                 capacity_limit, cam_resolution, simplify)
    #LPP.LHS
    #LPP.RHS
    #LPP.eLHS
    #LPP.eRHS
    
    #data_time_end_LPP = data_time_start - timeit.default_timer()
    #start = timeit.default_timer()
    ##############################################################
    ##### Stochastic multicriteria acceptability analysis ########
    ##############################################################
    
    
    #1# create different weights
        
    #2# get electre iii score for all sets of w
    #2# get topsis score for all sets of w
    # take average for all sets of scores
    
    #3# insert score in LPP an find solution
    #3# analyse relationship between weight input and solution
    
    # how often is each alternative chosen 
    # when alternative is chosen what is the average weight distribution? 
    # based on certain weight dis input - what is closest schedule(s)
    
    scenario.LPP = LPP
    scenario.df = DF_i
    scenario.pf_df = LPP.performance_df
    scenario.m = m
    return(EOSscenario)
    







def solve(x_data, scoring_method=2, solution_method="DAG", 
            criteria_weights = np.array([0,0,0,0,0,0,1,0]), 
            threshold_parameters= np.array([[0,0,1000],
                                            [0,0,40],
                                            [0,0,40],
                                            [0,0,15],
                                            [0,0,4],
                                            [0,0,20000],
                                            [0,0,1], 
                                            [0,0,1]]), 
            alpha = 1):
    #import ephem
    #from math import degrees, floor
    #import folium
    import pandas as pd
    import numpy as np
    import time
    #performance_df = x_data.pf_df
    #performance_df.info()
    #performance_df.columns[[7,11,12,13,15,16,17,18,19]]
    
    #### Call generated data frames 
    #performance_df.to_csv(r'C:\Users\allex\Desktop\SPECIALE\data\10reach.csv', index = ";", header=True)
    #read diff data 
    #reach10_data = pd.read_csv(r'C:\Users\allex\Desktop\SPECIALE\data\10reach.csv', sep = ",")
    #performance_df = reach10_data.iloc[:,1:]
    
    #naming convention
    LPP = x_data.LPP
    DF_i = x_data.df
    performance_df = x_data.pf_df
    
    #IDENTIFY CRITERIA TO INCLUDE IN SCORING PROCEDURE
    dat = np.array(performance_df.iloc[:,[7,11,12,13,15,17,18,19]].transpose())  #which criteria is important in performance df
    #NOTE FOR PAPER 3 - 16th col is not included as criteria 
    
    #from fuzzy_topsis import fuzzytopsis
    from EOSpython.easy_funcs import topsis
    from EOSpython.electre_parallel import parallelectre
    from cvxopt import matrix
    from cvxopt import glpk
    #glpk.options["show_progress"] = True
    #glpk.options["maxiters"] = 1000
    
    binVars = range(dat.shape[1])
    
    ###### setup test environment #######
    #scoring_method = 0 #0 = airbus, 1 = TOPSIS, 2 = ELECTRE, 3 = naive scoring method
    SMAA_version = 0 #0 = no usage, 1 = version 1 , 1 = version 2 
    MC_runs = 1
               #(area, distance, angle, sun elevation, cloud cover, pri, type, price, age, uncertainty)
# =============================================================================
#     q = np.array([1000,    10,      2,       1,           0,          0,    0,    0,    1,      0])    #indifference
#     p = np.array([1500,    25,      4,       5,           5,        0.8,    0.5,  3000,   5,   0.2])  #preferred
#     #v = np.array([20000,   100,    10,      10,          15,       0.3,    2,    40000, 6,  0.3])  #veto
#     v = np.array([100000, 250,    40,      70,           60,       1.2,    0.8, 100000, 10,   0.8])  #veto
# =============================================================================
    #####################################
    
    
    if SMAA_version == 2:
        #create weights
        MonteCarlo_runss = MC_runs
        weights = np.zeros((dat.shape[0], MonteCarlo_runss))
        i=0
        for k in list(np.array(list(range(1,1000, int(1000/(MonteCarlo_runss)))))/1000):
            weights[:,i] = np.random.dirichlet(np.ones(dat.shape[0])*float(k),size=1)
            i=i+1
        #np.std(weights, axis = 0)
    
    if SMAA_version == 1:
        MonteCarlo_runss = 1
    
    if SMAA_version == 0:
        MonteCarlo_runss = 1
        weights = np.zeros((dat.shape[0], MonteCarlo_runss))
        weights[:,0] = np.array([1]*dat.shape[0])/dat.shape[0] #equal
        #w_other = (1-0.5)/(dat.shape[0]-1)
        #customer type 0.5
        #weights[:,0] = np.array([w_other,w_other,w_other,w_other, 0.5, w_other,w_other,w_other])
        #uncertainty 0.5
        #weights[:,0] = np.array([w_other,w_other,w_other,w_other, w_other, w_other,w_other,0.5])
    
    #scenarios
    #(area, angle, sun elevation, cloud cover, pri, price, age, uncertainty)
    #weights[:,0] = np.array([0,0,0,0,0,0,0,1,0,0]) #profit
    #weights[:,0] = np.array([0,0,0,0,0,0,0,0,1,0]) #lead time
    #weights[:,0] = np.array([0,0,0,0,1,0,0,0,0,0]) #cloud cover
    #weights[:,0] = np.array([0.25, 0, 0, 0, 0.25, 0.25, 0.25, 0, 0, 0]) 
    if type(threshold_parameters) is not np.ndarray:
        q = np.array([0, 2, 2, 0, 0,0,0,0])
        p = np.array([50,5,5,5,1, 1000, 0, 2])
        v = np.array([1000, 40, 40, 15, 2, 10000, 13, 5])
# =============================================================================
#         The airbus mimiced parameters
#         q = np.array([0,0, 2, 2.424, 4.021, 0,0,0,1,0])
#         p = np.array([1500,0,4,5,5,0.8, 0.5, 3000, 5, 2])
#         v = np.array([100000, 0, 40, 70, 15.319, 0.812, 0.704, 100000, 10, 5])
# =============================================================================
    else:
        q = threshold_parameters[:,0]
        p = threshold_parameters[:,1]
        v = threshold_parameters[:,2]
    
    
    if type(criteria_weights) is not np.ndarray:
        weights[:,0] = np.array([1]*dat.shape[0])/dat.shape[0] #equal
    else:
        weights[:,0] = criteria_weights

    #weights[:,0] = np.array([0.0005, 0, 0.0004, 0.0002, 0.2885727, 0.13339262, 0.87803468, 0, 0.0005, 0])    
    objective_np = np.array([1,0,1,0,0,0,1,1,1]) #note for paper 3 - remove customer critiera 6th
    
    schedules = np.zeros((performance_df.shape[0], MonteCarlo_runss))
    for i in range(0, MonteCarlo_runss):
        
        if SMAA_version == 1:
            MonteCarlo_runs_v1 = MC_runs
            score_v1 = np.zeros((performance_df.shape[0], MonteCarlo_runs_v1))
            #len_x = len(dat[4,:])
            for h in range(0, MonteCarlo_runs_v1):
                dat[3,:] = dat[3,:] + np.multiply(np.random.uniform(-1,1,1)*20, dat[8,:])  #len_x instead of 1
                dat[3,:][dat[3,:] > 100] = 100
                dat[3,:][dat[3,:] < 0] = 0
                #global score with topsis or electre
                if (scoring_method == 0):
                    #priority
                    priority_airbus = [100000000,1000000,10000,6600,3300,120,1]
                    weather_airbus = (100-dat[3,:])/100
                    s_1cell = 3600
                    score_airbus = np.zeros((dat.shape[1]))
                    for s in range(0,dat.shape[1]):
                        score_airbus[s] = priority_airbus[int(dat[4,s])-1] * (1+ 4*weather_airbus[s] + 2*(dat[0,s]/s_1cell))
                        score = score_airbus
                    
                if (scoring_method == 1):
                    FT = topsis(dat, objective_np, weights[:,i])
                    score = FT.score
                    print('scored!')
                    #note topsis can score a request with 0, if it is the worst global alternative..
                    score = score + 0.000001
                    
                if (scoring_method == 2):
                    FT = parallelectre(dat, q, p, v, objective_np, weights[:,i])
                    score = np.mean(FT.score, axis = 1)
                    print('scored!')
                
                if (scoring_method == 3):
                    objective = objective_np
                    w_n = np.zeros((dat.shape))
                    score = np.zeros((dat.shape[1]))
                    for j in range(0, dat.shape[0]):
                        w_n[j,:] = dat[j,:]/max(dat[j,:])
                        if objective[j] == 0:
                            w_n[j,:] = 1 - w_n[j,:]
                    for j in range(0,dat.shape[1]):
                        score[j] = w_n[:,j] @ weights[:,0]
                    print('scored!')
                #save v1 scoring
                score_v1[:,h] = score
            score = np.mean(score_v1, axis = 1)
                
        else:
            #global score with topsis, electre, naive
            
            #priority modification for airbus score
            
                
            if (scoring_method == 1):
                FT = topsis(dat, objective_np, weights[:,i])
                score = FT.score
                #note topsis can score a request with 0, if it is the worst global alternative..
                score = score + 0.000001
                
            if (scoring_method == 2):
                FT = parallelectre(dat, p, q, v, objective_np, weights[:,i])
                score = np.mean(FT.score, axis = 1)
            
            if (scoring_method == 3):
                objective = objective_np
                w_n = np.zeros((dat.shape))
                score = np.zeros((dat.shape[1]))
                for j in range(0, dat.shape[0]):
                    w_n[j,:] = dat[j,:]/max(dat[j,:])
                    if objective[j] == 0:
                        w_n[j,:] = 1 - w_n[j,:]
                for j in range(0,dat.shape[1]):
                    score[j] = w_n[:,j] @ weights[:,i]
            
            #political valuation
            score = score**alpha
            print('scoring complete')
            
                
                
                
        if solution_method == "gurobi":
            import gurobipy as grb
            opt_model = grb.Model(name="BLP_Model")
            #LPP solution
            f = -score
            
            set_I = range(1, len(f)+1)
            # if x is Binary
            x_vars  = {(i): opt_model.addVar(vtype=grb.GRB.BINARY,
                       name="x_{0}".format(i)) for i in set_I}
            
            # <= constraints
            set_J = range(1, len(LPP.RHS)+1)
            a = {(j,i) : np.array(matrix(LPP.LHS))[j-1,i-1] for j in set_J for i in set_I}
            b = {(j): LPP.RHS[j-1] for j in set_J}
            c = {(j,i) : LPP.eLHS[j-1,i-1] for j in range(1,LPP.eLHS.shape[0]) for i in set_I}
            d = {(j): LPP.eRHS[j-1] for j in range(1,LPP.eLHS.shape[0])}
            
            #LPP.eLHS
            #LPP.eRHS
            # <= constraints
            constraints1 = {j : 
            opt_model.addConstr(
                    lhs=grb.quicksum(a[j,i] * x_vars[i] for i in set_I),
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=b[j], 
                    name="constraint1_{0}".format(j))
                for j in set_J}
            
            # == constraints
            constraints2 = {j : 
            opt_model.addConstr(
                    lhs=grb.quicksum(c[j,i] * x_vars[i] for i in set_I),
                    sense=grb.GRB.EQUAL,
                    rhs=b[j], 
                    name="constraint2_{0}".format(j))
                for j in range(1,LPP.eLHS.shape[0])}
            
            
            #OBJECTIVE
            f = {(i): -score[i-1] for i in set_I}    
            objective = grb.quicksum(x_vars[i] * f[i] for i in set_I)
            
            # for minimization
            opt_model.ModelSense = grb.GRB.MINIMIZE
            opt_model.setObjective(objective)
            
            #solve
            opt_model.optimize()
            
            #assign solution
            opt_df = pd.DataFrame.from_dict(x_vars, orient="index", columns = ["variable_object"])
            opt_df.reset_index(inplace=True)
            
            opt_df["solution_value"] = opt_df["variable_object"].apply(lambda item: item.X)
            
            #save x
            schedules[:,i] = np.array(opt_df["solution_value"])
            
            opt_model
            print(opt_model.solve_details.time, opt_model.solve_details.status)
            
            
            
            
        
        
        if solution_method == "PuLP":
            import pulp as plp
            opt_model = plp.LpProblem(name="BLP_Model")
            
            #LPP solution
            f = -score
            
            set_I = range(1, len(f)+1)
            x_vars  = {(i): plp.LpVariable(cat=plp.LpBinary, name="x_{0}_1".format(i)) for i in set_I}
                
            
            # <= constraints
            set_J = range(1, len(LPP.RHS)+1)
            a = {(j,i) : np.array(matrix(LPP.LHS))[j-1,i-1] for j in set_J for i in set_I}
            b = {(j): LPP.RHS[j-1] for j in set_J}
            c = {(j,i) : LPP.eLHS[j-1,i-1] for j in range(1,LPP.eLHS.shape[0]) for i in set_I}
            d = {(j): LPP.eRHS[j-1] for j in range(1,LPP.eLHS.shape[0])}
            
            #LPP.eLHS
            #LPP.eRHS
            constraints1 = {j : opt_model.addConstraint(
                    plp.LpConstraint(
                                 e=plp.lpSum(a[j,i] * x_vars[i] for i in set_I),
                                 sense=plp.LpConstraintLE,
                                 rhs=b[j],
                                 name="constraint1_{0}".format(j)))
                           for j in set_J}
            
            # == constraints
            constraints2 = {j : opt_model.addConstraint(
            plp.LpConstraint(
                         e=plp.lpSum(c[j,i] * x_vars[i] for i in set_I),
                         sense=plp.LpConstraintEQ,
                         rhs=d[j],
                         name="constraint2_{0}".format(j)))
                   for j in range(1,LPP.eLHS.shape[0])}
            
            #OBJECTIVE
            f = {(i): -score[i-1] for i in set_I}    
            objective = plp.lpSum(x_vars[i] * f[i] for i in set_I)
            
            # for minimization
            opt_model.sense = plp.LpMinimize
            opt_model.setObjective(objective)
            
            # solving with CBC
            opt_model.solve()
            
            #assign solution
            import pandas as pd
            opt_df = pd.DataFrame.from_dict(x_vars, orient="index", columns = ["variable_object"])
            opt_df.reset_index(inplace=True)
            
            opt_df["solution_value"] = opt_df["variable_object"].apply(lambda item: item.varValue)
            
            
            #save x
            schedules[:,i] = np.array(opt_df["solution_value"])

            print(plp.LpStatus[opt_model.status])
       
        
        if solution_method == "cplex":
            import docplex.mp.model as cpx
            opt_model = cpx.Model(name="BLP_Model")
            #LPP solution
            f = -score
            
            set_I = range(1, len(f)+1)
            x_vars = {(i,0): opt_model.binary_var(name="x_{0}_0".format(i)) for i in set_I}
            
            # <= constraints
            set_J = range(1, len(LPP.RHS)+1)
            a = {(j,i) : np.array(matrix(LPP.LHS))[j-1,i-1] for j in set_J for i in set_I}
            b = {(j): LPP.RHS[j-1] for j in set_J}
            c = {(j,i) : LPP.eLHS[j-1,i-1] for j in range(1,LPP.eLHS.shape[0]) for i in set_I}
            d = {(j): LPP.eRHS[j-1] for j in range(1,LPP.eLHS.shape[0])}
            
            #LPP.eLHS
            #LPP.eRHS
            constraints1 = {j : 
            opt_model.add_constraint(
                    ct=opt_model.sum(a[j,i] * x_vars[i,0] for i in set_I) <= b[j],
                    ctname="constraint1_{0}".format(j)) for j in set_J}
            
            # == constraints
            constraints2 = {j : 
            opt_model.add_constraint(
                    ct=opt_model.sum(c[j,i] * x_vars[i,0] for i in set_I) == d[j],
                    ctname="constraint2_{0}".format(j)) for j in range(1,LPP.eLHS.shape[0])}
            
            #OBJECTIVE
            f = {(i): -score[i-1] for i in set_I}    
            objective = opt_model.sum(x_vars[i,0] * f[i] 
                          for i in set_I)
            
            #solve with cplex cloud
            opt_model.time_limit = 60 #1e75          

            API_cplexcloud = 'api_a60ae489-9091-4c39-b7e2-90135566c662'
            cplex_cloud_url = 'https://api-oaas.docloud.ibmcloud.com/job_manager/rest/v1/'
            opt_model.solve(url=cplex_cloud_url, key=API_cplexcloud)
            
            #assign solution
            opt_df = pd.DataFrame.from_dict(x_vars, orient="index", columns = ["variable_object"])
            opt_df.index = pd.MultiIndex.from_tuples(opt_df.index, names=["column_i", "column_j"])
            opt_df.reset_index(inplace=True)
            
            opt_df["solution_value"] = opt_df["variable_object"].apply(lambda item: item.solution_value)
            
            #save x
            schedules[:,i] = np.array(opt_df["solution_value"])

            print(opt_model.solve_details.time, opt_model.solve_details.status)
                        
            
        if solution_method == "GLPK":
            #LPP solution
            f = -score
            start_time = time.time()

            output_cvxopt = glpk.ilp(c = matrix(f), G = LPP.LHS, h = matrix(LPP.RHS), # A = matrix(LPP.eLHS), b = matrix(LPP.eRHS),
                                     I = set(binVars), B = set(binVars))
            x = np.array(output_cvxopt[1])
            
            end_time = time.time() - start_time
            
            schedules[:,i] = np.squeeze(x)
    
            print(f@x, end_time)
            #global airbus score  
            #GS = np.dot(np.squeeze(x), score_airbus)
            #f@x
            #np.sum(x)/len(x)
            
            
        if solution_method[0:3] == "DAG":
            #note, currently all attempts are denoted by a digit - we do another abstraction 
            #to represent each choice as the edge between two attempts, that is each attempt 
            #is a node, while all the feasible rotations between attempts are the edges of  
            #the graph. All edges are represented by the inverted infeasibility matrix.
            #NOW SORT the NODES after satellite and time:
            #performance_df.info() 
            
            
            ##TOPOLOOGICAL SORTING - relative to satellite and time
            new_index_list = list(range(0,len(performance_df)))
            for i in range(0,len(performance_df)-1):
                if not (performance_df.iloc[i]["satellite"] == performance_df.iloc[i+1]["satellite"] and performance_df.iloc[i]["time"] <= performance_df.iloc[i+1]["time"]):
                    #something is maybe wrong with topological sorting
                    if (performance_df.iloc[i]["time"] >= performance_df.iloc[i+1]["time"] and performance_df.iloc[i]["satellite"] == performance_df.iloc[i+1]["satellite"]):
                        #interchange index i and i+1
                        new_index_list[i+1] = i
                        new_index_list[i] = i+1
            
            #check for changes needed
            numberofsatellites = 2
            check_1 = any(np.diff(new_index_list) == 0) 
            check_2 = np.sum(np.diff(np.array(performance_df['satellite']))!=0) != numberofsatellites-1 #number of satellites
            #check_3 = any(np.array(np.diff(np.array(performance_df['time'], dtype = 'timedelta64[ns]')), dtype = 'int64')<0)
            if check_1 or check_2: # or check_3:
                #if changes are required modify performance_df, score, F, etc. - unlikely as this sorting should have been done prior.
                print('topological sorting error with', numberofsatellites, 'satellites')            
                            
            
            #create edges matrix (no loops)
            edges0 = LPP.F==0
            edges = np.triu(edges0, 1) #remove loops and directed (cannot go backwards)           
            
            #Algorithm should not include interdependent sets, that is, attempts representing the same request 
            #inter_attempts = LPP.B
            
            
            
            #stereo and strips interdependencies
            
            #stereo
            stereo = LPP.stereo
            
            #strips 
            
            all_strip_reqs_id = performance_df['ID'].iloc[np.where(performance_df['strips']>=1)]
            unique_strip_reqs = np.unique(all_strip_reqs_id)
            strips = np.zeros((len(unique_strip_reqs), len(performance_df)))
            strips_num_acq = list()
            for ii in range(0,len(unique_strip_reqs)):
                index_i = all_strip_reqs_id.iloc[np.where(all_strip_reqs_id == unique_strip_reqs[ii])[0]].index
                strips[ii, index_i] = 1
                strips_num_acq.append(max(performance_df[['stereo','strips']].iloc[index_i[0]]+np.array([1,0])))
            
            
            depth_edges_Search = solution_method[3:]
            if depth_edges_Search == '':
                depth_edges_Search = 25
            else:
                depth_edges_Search = int(depth_edges_Search)
            
            start_time = time.time()
            
            #remove function
            def REMOVE(path, I, edges, stereo, strips, strips_num_acq,score):
                path[I] = 0
                if any(path) == False:
                    return(path)
                else:
                    number_before_I = int(np.sum(path[0:I])) 
                    if number_before_I > 0:
                        from_a = np.where(path)[0][number_before_I-1] #the number before variable starts from 1
                    else:
                        from_a = 0
                    number_after_I = int(np.sum(path[I:]))
                    if number_after_I > 0:
                        from_b = np.where(path)[0][number_before_I]
                    else:
                        return(path)
                        
                    search_range = [from_a, from_b]    
                    
                    into=np.where(edges[search_range[0],:])[0]
                    feasible_nodes = into[np.where(edges[into, search_range[1]])[0]]
                    feasible_nodes = np.setdiff1d(feasible_nodes, I)
                    if len(feasible_nodes) == 0:
                        return(path)
                    else:
                        longest_path_tempp = [[] for ii in range(0,len(feasible_nodes))]
                        weight_of_path_tempp = np.zeros((len(feasible_nodes)))
                        for iii in range(0,len(feasible_nodes)):
                            path_temp = np.copy(path)
                            path_temp[feasible_nodes[iii]] = 1
                            if all(strips @ path_temp <= strips_num_acq) and all(stereo @ path_temp == 0) :
                                longest_path_tempp[iii] = path_temp
                                weight_of_path_tempp[iii] = score[feasible_nodes[iii]]
                        if np.max(weight_of_path_tempp) == 0:
                            return(path)
                        else:
                            return(longest_path_tempp[np.argmax(weight_of_path_tempp)])
            
            #insert node in path function 
            def INSERT(x, I, keep, edges, stereo, strips, strips_num_acq,score):
                x[I] = 1
                if any(strips @ x > strips_num_acq):
                    #remove least contributing interdep node
                    interdep = np.where(strips[np.where(strips[:,I])[0],:])[1] 
                    interdep_i_acq = np.intersect1d(interdep,np.where(x)[0])
                    if keep != []:
                        interdep_i_acq = np.setdiff1d(interdep_i_acq,keep)
                    interdep = np.delete(interdep_i_acq,np.where(interdep_i_acq==I)[0])  #single
                    x = REMOVE(x, interdep[np.argmin(score[interdep])], edges, stereo, strips, strips_num_acq, score) 
                
                if any(stereo @ x != 0):
                    stereo_idx = np.where(stereo[np.where(stereo @ x != 0)[0], :]==1)[1] #single
                    x = REMOVE(x, stereo_idx[np.argmin(score[stereo_idx])], edges, stereo, strips, strips_num_acq, score)
                
                r1 = np.squeeze(edges[np.where(x[0:I])[0],I])
                r2 = np.squeeze(edges[I,np.where(x[I:])[0] + I])
                feasible_maneuver = np.concatenate((r1.reshape((r1.size)), r2.reshape((r2.size))))
                if np.sum(feasible_maneuver == False) > 1: #because I is infeasible with it self  
                    infeas = np.where(x)[0][np.where((np.where(x)[0] != I)*1 + ~feasible_maneuver.astype(bool) == 2)[0]]
                    
                    if len(infeas) < 2:
                        x = REMOVE(x, infeas[0], edges, stereo, strips, strips_num_acq, score)
                    else:
                        infeas_lims = infeas[[0,len(infeas)-1]]
                        x[infeas] = 0
                        for infs in infeas_lims:
                            x = REMOVE(x, infs, edges, stereo, strips, strips_num_acq, score) 
                    
                return(x)
            
            #EXAMPLE from paper:
#            edges = np.array([[0,1,1,1,1,1,1,1,1,1],
#                              [0,0,1,1,1,1,1,1,1,1],
#                              [0,0,0,0,1,1,1,1,1,1],
#                              [0,0,0,0,1,1,1,1,1,1],
#                              [0,0,0,0,0,1,1,1,1,1],
#                              [0,0,0,0,0,0,1,1,1,1],
#                              [0,0,0,0,0,0,0,1,1,1],
#                              [0,0,0,0,0,0,0,0,0,1],
#                              [0,0,0,0,0,0,0,0,0,1],
#                              [0,0,0,0,0,0,0,0,0,0]])
#            strips =np.array([[1,0,0,0,0,1,0,0,0,0],
#                              [0,1,1,0,0,0,1,1,0,0],
#                              [0,0,0,1,1,0,0,0,1,1]])
#            strips_num_acq = np.array([1,2,2])
#            stereo = np.array([[0,0,0,1,0,0,0,0,-1,0],
#                               [0,0,0,0,1,0,0,0,0,-1]])
#            score = np.array([1,1,2,3,2,2,2,2,2,3])
            
            #x = [0, 0, 1, 0, 0, 1, 1, 0, 1, 0]
            #I=3
            #INSERT(path, I, edges, stereo, strips, strips_num_acq)
            
            ##EXTENDED LONGEST PATH ALGORITHM    
            longest_path_to_node = [[] for i in range(0,len(edges))]
            weight_of_path = np.zeros((len(edges)))
                    
            for i in range(0, len(edges)):
                incomming_neighbours = list(np.where(edges[0:i,i])[0])
                #if zero longest path is just it se lf.    
                if len(incomming_neighbours) == 0:
                    if all(stereo[:,i] == 0): #non-stereo
                        #longest_path_to_node[i].append(-1)
                        longest_path_to_node[i].append(i)
                        weight_of_path[i] = score[i]
                    else:
                        weight_of_path[i] = 0
                else:
                    ##initiate loop to find largest path not including an interdependent node
                    sort_neighbour = np.argsort(weight_of_path[incomming_neighbours])[::-1]
                    vertice_which = np.array(incomming_neighbours)[sort_neighbour]
                    #naive shorting of incoming neighbours - parameter to how deep it should investigate
                    depth = min(depth_edges_Search,len(incomming_neighbours))
                    longest_path_temp = [[] for jj in range(0,depth)]
                    weight_of_path_temp = np.zeros((depth))
                    for j in range(0,depth):
                        path = np.concatenate((longest_path_to_node[vertice_which[j]], [i]))
                        x = np.zeros((len(edges)))
                        x[path.astype(int)] = 1
                        if all(strips @ x<= strips_num_acq):
                            if all(stereo @ x == 0):
                                if j == 0:
                                    longest_path_to_node[i] = np.where(x)[0]
                                    weight_of_path[i] = score @ x
                                    break
                                else:
                                    longest_path_temp[j] = x
                                    weight_of_path_temp[j] = score @ x
                            else:
                                #ADD stereo or path is not possible
                                stereo_set = np.where(stereo[np.where(stereo[:,i] == -1)[0],:] == 1)[1]
                                if len(stereo_set) == 0:
                                    weight_of_path_temp[j] = 0
                                    continue
                                else:
                                    #print(i,j,x,path,stereo_set)
                                    x_stereo = INSERT(x, stereo_set[0], i, edges, stereo, strips, strips_num_acq,score)
                                    longest_path_temp[j] = x_stereo
                                    weight_of_path_temp[j] = score @ x_stereo
                        else:
                            #remove least contributing interdep node
                            if all(stereo[:,i] == 0): #non-stereo
                                np.intersect1d(np.where(strips[np.where(strips @ x > strips_num_acq)[0],:])[1], np.where(x[:i])[0])
                                interdep_i = np.where(strips[np.where(strips[:,i])[0],:])[1] 
                                interdep_i_acq = np.intersect1d(interdep_i,np.where(x)[0])
                                interdep_i_acq = np.setdiff1d(interdep_i_acq, i)
                                
                                min_interdep_i = np.argmin(score[interdep_i_acq])
                                min_ilegal_node = interdep_i_acq[min_interdep_i]
                                longest_path_temp[j] = REMOVE(x, min_ilegal_node, edges, stereo, strips, strips_num_acq,score)
                            else:
                                ind_stereo = np.where(stereo[np.where(stereo[:,i]==-1)[0],:i] == 1)[1]
                                if len(ind_stereo) == 0:
                                    #legal path to this one is not feasible
                                    weight_of_path_temp[j] = 0
                                    continue
                                else:
                                    #we now know that another set of stereo attempts are performed of the same stereo request as node i is trying to acquire
                                    #so that pair has to be terminated and the not-included should be included:
                                    #np.where(performance_df['index'][i] == performance_df['index'])[0]
                                    stereo_pairs = np.intersect1d(ind_stereo, path) 
                                    if len(stereo_pairs) == 0: #the other pair is not included
                                        int_stereo_set = np.intersect1d(np.where(strips[np.where(strips @ x > strips_num_acq)[0],:])[1], np.where(x[:i])[0])
                                        x[int_stereo_set] = 0 #investigated after adding the others
                                        ##
                                        if len(ind_stereo) > 1: #mistake if this can happen
                                            print(ind_stereo)
                                        ##
                                        
                                        new_stereo = int(ind_stereo)
                                        #x_temp2 = np.zeros(len(x))
                                        #length_temp2 = np.zeros(len(ind_stereo))
                                        #for i_ss in range(0,len(ind_stereo)):
                                        x = INSERT(x, new_stereo, i, edges, stereo, strips, strips_num_acq, score)
                                        #length_temp2 = score @ x_temp2
                                        #x = x_temp2[:,length_temp2]    
                                        for i_s in int_stereo_set:
                                            x = REMOVE(x, i_s, edges, stereo, strips, strips_num_acq, score) #investigates the first of the prior removed stereo attempt
                                        longest_path_temp[j] = x
                                    else:
                                        min_stereo = stereo_pairs[np.argmin(score[stereo_pairs])] #modified
                                        longest_path_temp[j] = REMOVE(x, min_stereo, edges, stereo, strips, strips_num_acq,score)
                            weight_of_path_temp[j] = longest_path_temp[j] @ score
                    
                    if len(longest_path_to_node[i]) == 0:
                        longest_path_to_node[i] = np.where(longest_path_temp[np.argmax(weight_of_path_temp)])[0]
                        weight_of_path[i] = np.max(weight_of_path_temp)
                        

            x = np.zeros((len(edges)))
            x[longest_path_to_node[np.argmax(weight_of_path)]] = 1
            
            #x = longest_path_to_node
            
            end_time = time.time() - start_time
            #schedules[:,i] = np.squeeze(x)
            print('acq, objvalue, runtime:', np.sum(x),  -np.max(weight_of_path), end_time)                    
                    
#            ##EXTENDED LONGEST PATH ALGORITHM    
#            longest_path_to_node = [[] for i in range(0,len(performance_df))]
#            weight_of_path = np.zeros((len(performance_df)))
#
#            
#            for i in range(0, len(performance_df)):
#                #i+=1
#                incomming_neighbours = list(np.where(edges[max(0,i-depth_edges_Search):i,i])[0])
#                #if zero longest path is just it se lf.    
#                if len(incomming_neighbours) == 0:
#                    #longest_path_to_node[i].append(-1)
#                    longest_path_to_node[i].append(i)
#                    weight_of_path[i] = score[i]
#                else:
#                    #find all interdependent attempts relative to current node
#                    which_id = np.where(inter_attempts[:,i]==1)[0]
#                    inter_i = np.where(inter_attempts[which_id,:])[1]
#                    #relative to strips and stereo
#                    allowed_stereo = set(np.where(stereo[np.where(stereo[:,i] == -1)[0],:i])[0])
#                    strips_interdependent_i = np.where(strips[:,i]==1)[0]
#                    
#                    
#                    ##initiate loop to find largest path not including an interdependent node
#                    max_path_weight = 0
#                    max_path = list()
#                    for j in range(0,min(depth_independent_Search,len(incomming_neighbours))):
#                        max_neighbour = np.argmax(weight_of_path[incomming_neighbours])
#                        vertice_which = incomming_neighbours[max_neighbour]
#                        #check if vertice is already included (interdependent other node)
#                        s1 = set(longest_path_to_node[vertice_which])
#                        s2 = set(inter_i)
#                        intersection = s1.intersection(s2)
#                        if j == 0 and len(intersection) == 0:
#                            max_path = list(s1)
#                            break
#                        #check if any interdependent can be omitted due to the stereo and strip allowing constraints?
#                        #  Note, intersection is the similarity between interdependent attempts and already included relative to the current investigated attempt.
#                        #  If we remove attempts from intersection, they are removed from interdependent list, and thereby added to the final list checked for max path.
#                        
#                        #stereo - just remove from intersection if they are allowed by stereo constraint.
#                        intersection = intersection - allowed_stereo
#                        
#                        #strips
#                        if len(strips_interdependent_i) > 0: 
#                            strips_interdependent = set(np.where(strips[strips_interdependent_i,:])[1])
#                            strips_allowed = list(intersection.intersection(strips_interdependent))
#                            #all strips are allowed, if number of strips does not exceed the allowed constrained number.
#                            if len(strips_allowed) < strips_num_acq[int(strips_interdependent_i)]:
#                                intersection = intersection - set(strips_allowed)
#                            #locate least benefitting attempt in strips (interdepedent) - remove that! (note, neglects possible profit from alternative where old attempt where removed)
#                            else: 
#                                min_strip_i = np.argmin(weight_of_path[strips_allowed])
#                                min_strip = strips_allowed[min_strip_i]
#                                #remove the least contributing strip from intersec
#                                intersection = intersection - (set(strips_allowed) - set([min_strip]))
#                                s1 = s1 - set([min_strip])
#                        #As a rule of thumb the intersection will at most include one illegal node.
#                        #we therefore search if there is any legal nodes that connects the same pair 
#                        #of nodes as the intersection, if so we include that 
#                        #if len(s1) > 1 and len(intersection)==1:
#                        #    s1.add(i)
#                        #    s1_check = np.array(list(s1))
#                        #    argsort = np.argsort(np.abs(s1_check-np.array(list(intersection))))
#                        #    idx_min = np.where(np.isin(argsort,[1,2]))
#                        #    s1_id = s1_check[idx_min]
#                        #    s1.remove(i)
#                        #    #find other connecting edge:
#                        #    connecting0 = np.where(1*edges[s1_id[0],s1_id[0]:(s1_id[1]+1)] + 1*edges[s1_id[0]:(s1_id[1]+1), s1_id[1]] == 2)[0]
#                        #    connecting = np.array(range(s1_id[0], s1_id[1]+1))[connecting0]
#                        #    
#                        #    #check if not already included 
#                        #    not_included = np.where(np.isin(connecting, list(s1)) == False)[0]
#                        #    if len(not_included) > 0:
#                        #        #continue check and add node to that s1
#                        #        connecting = connecting[not_included]
#                        #        #legal?
#                        #        legal = connecting[np.where(np.isin(connecting, strips_interdependent_i) == False)[0]]
#                        #        #highest
#                        #        highest_connection = int(legal[np.argmax(score[legal])])
#                        #        s1.add(highest_connection)
#                        #        #Note, should not be added for future nodes as only relevant for this one
#                
#                            #if no edges connects - path is as it is
#                            
#                        #path weight without previous interdependent node
#                        max_path0 = list(s1-intersection)
#                        max_path_weight0 = np.sum(score[max_path0])
#                        if max_path_weight0 > max_path_weight:
#                            max_path = max_path0
#                            max_path_weight = max_path_weight0
#                        #delete the already checked vertice
#                        del incomming_neighbours[max_neighbour]
#                        
#                    longest_path_to_node[i] = max_path  
#                    longest_path_to_node[i].append(i)
#                    weight_of_path[i] = np.sum(score[longest_path_to_node[i]])  
#                    
#                    
#            x = np.zeros((len(performance_df)))
#            x[longest_path_to_node[np.argmax(weight_of_path)]] = 1
#            
#            #x = longest_path_to_node
#            
#            end_time = time.time() - start_time
#            #schedules[:,i] = np.squeeze(x)
#            print(depth_independent_Search, depth_edges_Search,  np.max(weight_of_path), end_time)
                        
            
            #SMAA
            
            
            
        if solution_method == "VNS":
            #object fct
            c = score
            #<=
            G = np.array(matrix(LPP.LHS)) 
            h = LPP.RHS 
            #=
            A = LPP.eLHS
            b = LPP.eRHS
            #number of runs 
            max_time = 120 #sec
            max_iter_N = 1000  #per neighbourhood

            
            #define neighbourhoods
            string_list = list()
            BBB = list(performance_df["request location"])
            string_list = list(set(list(map(str, BBB))))
            number_of_reach_areas = len(string_list)
            import ast
            string_list_np = np.array([ast.literal_eval(n) for n in string_list])
            pfloc_np = np.array([xi for xi in performance_df["request location"]])
            N_i = np.zeros((number_of_reach_areas, len(pfloc_np)))
            N_i_rhs = np.zeros((number_of_reach_areas))
            for i in range(0,number_of_reach_areas):
                N_i[i,:] = np.sum((pfloc_np == string_list_np[i,:]), axis = 1) == 2
                one_index_reach = np.where(np.sum(pfloc_np == string_list_np[i], axis = 1) == 2)[0][0]
                N_i_rhs[i] = max(performance_df["stereo"][one_index_reach], performance_df["strips"][one_index_reach])
            
            
            
            #begin requestbased neighbourhood search
            from random import randint
            import datetime
            VNS_time_max = datetime.datetime.now() + datetime.timedelta(seconds = max_time)
            fx_min = 0
            x_i = np.zeros((c.shape[0]))
            x_opt = np.zeros((c.shape[0]))
            N = 0
            while N < N_i.shape[0] and VNS_time_max > datetime.datetime.now():
                N_loc = np.where(N_i[N,:])[0]
                N_size = len(N_loc)
                x_i = list(x_opt)
                for i in range(0,max_iter_N):
                    no_ones = randint(1,min(N_i_rhs[N],N_size))
                    random_x = np.array([1] * no_ones + [0] * int(N_size-no_ones))
                    np.random.shuffle(random_x)
                    x_i = np.array(x_i)
                    x_i[N_loc] = random_x
                    fx_i = c@x_i
                    if ((G@x_i <= h).all() and (A@x_i == b).all() and fx_i < fx_min):
                        fx_min = fx_i
                        x_opt = list(x_i)
                        N = -1
                        print(fx_min)
                        break
                
            x = np.array(x_opt)
            
        
        if solution_method == "random":
            start_time = time.time()
            #generate 2000 random solutions and test objective function for each
            #choose interval that illustrates distribution of ones and zeros 
            N = 1000
            binary_int = [0.03,0.18]
            binary_int_dist = binary_int[1]-binary_int[0]
            len_p = len(performance_df)
            X_test = np.zeros((len_p, N))
            for i in range(0, N):
                dist_1 = binary_int[0] + i/N * (binary_int_dist)
                X_test[:,i] = np.random.choice([0, 1], size=(len_p,), p=[1-dist_1, dist_1])
            
            #test if the solutions are valid
            valid_x = np.zeros((N))
            for i in range(0,N):
                if (all(np.array(matrix(LPP.LHS))@X_test[:,i] <= LPP.RHS) and all(np.array(matrix(LPP.eLHS))@X_test[:,i] == LPP.eRHS)):
                    valid_x[i] = 1
                    
            #find maximal score
            sum_valid = int(np.sum(valid_x))
            if sum_valid > 0:
                score_test = np.transpose(np.ones((sum_valid, len(score))) * score)
                score_sum = np.sum(X_test[:,np.where(valid_x)[0]]*score_test, axis = 0)
                max_i = np.argmax(score_sum)
                x = X_test[:,np.where(valid_x)[0]][:,max_i]
                print(score_sum[max_i])
            else:
                x = np.zeros(len_p)
                print('problem too complex')
            
            end_time = time.time() - start_time
            print(end_time)
        
        
        
        solve.x = np.squeeze(x)
        solve.score = score
        solve.time = end_time
        return(solve)
    
   
 
def visualize(x_data, x_res, name_of_html = 'EOSpython', color = 'black'):
    #plot multi sat scenario and solution
    import numpy as np
    import folium
    #import copy
    performance_df = x_data.pf_df
    schedules = x_res.x
    a = x_data.m
    idx_acqs = np.where(schedules)[0]
    points_entire = list()
    
    for i in range(0,int(np.sum(schedules))):
        index_line = idx_acqs[i]
        points = list()
        points.append(tuple(performance_df.iloc[index_line, 4]))
        points.append(tuple(performance_df.iloc[index_line, 5])) 
        #print(index_line, points)
        
        points_entire.append(points)
        
    folium.PolyLine(points_entire, color=color, weight=2.5, opacity=1).add_to(a)
    
    a.save(name_of_html + ".html")


def evaluate(x_data, x_res):
    import numpy as np
    import pandas as pd
    
    #scenario generation specific evalation metrics
    reqss_m = int(len(np.unique(x_data.pf_df['ID'])))
    atts_m = int(len(x_data.pf_df))
    constraintss_m = int(x_data.LPP.eRHS.shape[0] + x_data.LPP.RHS.shape[0])
    angles_m = np.mean(x_data.pf_df['angle'])
    areas_m = np.mean(x_data.pf_df['area'])
    prices_m = np.mean(x_data.pf_df['price'])
    sun_elevations_m = np.mean(x_data.pf_df['sun elevation'])
    ccs_m = np.mean(x_data.pf_df['cloud cover real'])
    prio_m = np.mean(x_data.pf_df['priority'])
    
    
    #solution specific metrics
    acq = int(np.sum(x_res.x))
    profit = np.sum(x_data.pf_df['price'].iloc[np.where(x_res.x)])
    avg_cloud = np.mean(x_data.pf_df['cloud cover real'].iloc[np.where(x_res.x)])
    cloud_good = int(np.sum(x_data.pf_df['cloud cover real'].iloc[np.where(x_res.x)]<10))
    cloud_bad = int(np.sum(x_data.pf_df['cloud cover real'].iloc[np.where(x_res.x)]>30))
    avg_angle = np.mean(x_data.pf_df['angle'].iloc[np.where(x_res.x)])
    angle_good = int(np.sum(x_data.pf_df['angle'].iloc[np.where(x_res.x)]<=10))
    angle_bad = int(np.sum(x_data.pf_df['angle'].iloc[np.where(x_res.x)]>=30))
    avg_priority = np.mean(x_data.pf_df['priority'].iloc[np.where(x_res.x)])
    priority1 = int(np.sum(x_data.pf_df['priority'].iloc[np.where(x_res.x)]==1))
    priority2 = int(np.sum(x_data.pf_df['priority'].iloc[np.where(x_res.x)]==2))
    priority3 = int(np.sum(x_data.pf_df['priority'].iloc[np.where(x_res.x)]==3))
    priority4 = int(np.sum(x_data.pf_df['priority'].iloc[np.where(x_res.x)]==4))
    sunelevation = np.mean(x_data.pf_df['sun elevation'].iloc[np.where(x_res.x)])
    totalarea =  np.sum(x_data.pf_df['area'].iloc[np.where(x_res.x)])
    
    EOSevaluate.scenario = pd.DataFrame({'metric':['requests', 'attempts','constraints', 'avg angle', 'avg area', 'avg price', 'avg sun elevation', 'avg cloud cover', 'avg priority'],
                                 'value':[reqss_m, atts_m, constraintss_m, angles_m, areas_m, prices_m, sun_elevations_m, ccs_m, prio_m]})


    EOSevaluate.solution = pd.DataFrame({'metric':['acquisitions', 'total profit', 'avg cloud cover', 'cloud cover < 10', 'cloud cover > 30', 'avg angle', 'angle < 10', 'angle > 30', 'avg priority', 'priority 1', 'priority 2', 'priority 3', 'priority 4', 'avg sun elevation', 'total area'],
                                 'value':[acq, profit, avg_cloud, cloud_good, cloud_bad, avg_angle, angle_good, angle_bad, avg_priority, priority1, priority2, priority3, priority4, sunelevation, totalarea]})

    return(EOSevaluate)
