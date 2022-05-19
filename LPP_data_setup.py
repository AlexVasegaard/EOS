#LPP data computation - multi satellite (no battery, higher dim location_slots, etc)
def LPP_data_multi(performance_df, number_of_acq_points, time_slots, location_slots, 
             height_satellite, rotation_speed, seconds_gran,
             capacity_limit, cam_resolution, simplify):
    
    #packages
    import numpy as np
    import progressbar

    bar = progressbar.ProgressBar(max_value=len(location_slots)*performance_df.shape[0]+7)
    bar_i = 1
    bar.update(bar_i)
    
    
    
    
    len_x = performance_df.shape[0]
    #functions
    #conversion into cartesian coordinate system. Note R is appr. radius of earth 6371 km
    def cart_system(lat, lon, elevation):
        import numpy as np 
        lat = np.radians(lat)
        lon = np.radians(lon)
        R = 6371 + elevation
        x = R * np.cos(lat) * np.cos(lon)
        y = R * np.cos(lat) * np.sin(lon)
        z = R * np.sin(lat)
        return(np.array([x,y,z]))
    #identify doubles
    def insert_row(idx, df, df_insert):
        dfA = df.iloc[:idx, ]
        dfB = df.iloc[idx:, ]
        df = dfA.append(df_insert).append(dfB).reset_index(drop = True)
        return(df)
    #repeat element n times 
    def repeatelem(elem, n):
        #returns an array with element elem repeated n times.
        arr = np.array([])
        if n == 0:
            return([])
        else:
            for i in range(n):
                arr = np.concatenate((arr,elem))        
                arr
        return(arr)
    
    
    #### stereo requests  - first as we modify the performance df
    #with which angle must 3D images be acquired
    stereo_angle = 17.5
    stereo_error = 2.5
    stereo_perf_df = np.where(performance_df["stereo"]>0)[0]
    stereo_IDs = list(np.unique(performance_df["ID"][stereo_perf_df]))
    S = list()
    for i in range(0,len(stereo_IDs)):
        range_stereo = list(np.where(performance_df["ID"] == stereo_IDs[i])[0])
        if len(range_stereo) <= 1:
            continue
        for a in range(0,len(range_stereo)-1):
            sat_xyz1 = cart_system(performance_df["satellite location"][range_stereo[a]][0],performance_df["satellite location"][range_stereo[a]][1],height_satellite)
            loc_xyz1 = cart_system(performance_df["request location"][range_stereo[a]][0],performance_df["request location"][range_stereo[a]][1],0)
            vec1 = sat_xyz1-loc_xyz1  #inverse vector: from req to sat
            for b in range(a+1, len(range_stereo)):
                sat_xyz2 = cart_system(performance_df["satellite location"][range_stereo[b]][0],performance_df["satellite location"][range_stereo[b]][1],height_satellite)
                loc_xyz2 = cart_system(performance_df["request location"][range_stereo[b]][0],performance_df["request location"][range_stereo[b]][1],0)
                vec2 = sat_xyz2-loc_xyz2 #inverse vector: from req to sat
                cos_radia = np.min([np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), 1]) #rounding error
                angle = np.degrees(np.arccos(cos_radia))
                
                if (angle >= stereo_angle - stereo_error and angle <= stereo_angle + stereo_error):
                    S.append([range_stereo[a],range_stereo[b]])
    
    #pairwise representation
    S_constraint = np.zeros((len(S), len_x))
    for i in range(0,len(S)):
        S_constraint[i,S[i][0]] = 1
        S_constraint[i,S[i][1]] = -1


#    S_constraint = np.array([[1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0],
#                             [0,1,-1,0,0,0,0,0,0,0,0,0,0,0,0],
#                             [0,0,1,0,-1,0,0,0,0,0,0,0,0,0,0],
#                             [1,0,0,0,0,0,-1,0,0,0,0,0,0,0,0]])

#    S_constraint = np.array([[1,-1,0,0,0],
#                             [1,0,-1,0,0],
#                             [0,1,-1,0,0]])
    
    stereos = np.sum(S_constraint!=0, axis = 0)
    while any(stereos > 1):
        observed_double = np.where(stereos>1)[0][0] #first observed double column
        observed_double_row = np.where(S_constraint[:,observed_double] != 0)[0][1]  #second observed dobule row in that column
        S_constraint = np.concatenate((S_constraint[:,:(observed_double+1)], np.zeros((S_constraint.shape[0],1)), S_constraint[:,(observed_double+1):]), axis = 1)
        #remove the remaining in that col and move to new column + update performance df and len x
        #performance df
        performance_df = insert_row(observed_double, performance_df, performance_df.iloc[observed_double,:])
        #replicate, delete, update len x
        S_constraint[observed_double_row:, observed_double+1] = S_constraint[observed_double_row:, observed_double]
        S_constraint[observed_double_row:, observed_double] = 0
        len_x = performance_df.shape[0]
        stereos = np.sum(S_constraint!=0, axis = 0)

#    stereos = np.sum(S_constraint!=0, axis = 0)
#    interest_ids = np.where(stereos>=2)[0]
#    stereo_doubles = np.array([])
#    for i in np.unique(stereos[interest_ids]):
#        stereo_doubles = np.append(stereo_doubles, repeatelem(np.where(stereos==i)[0],int(i-1)))
#    stereo_doubles = np.sort(stereo_doubles)
#    len_stereo_doubles = len(stereo_doubles) 
#    if len_stereo_doubles != 0:
#        #update performance_df (replicate the double nodes), the stereo constraint function, and the len_x call..
#        S_constraint = np.concatenate((S_constraint,np.zeros((S_constraint.shape[0],len_stereo_doubles))), axis = 1)
#        
#        for s in range(0,len_stereo_doubles):
#            idx = int(stereo_doubles[s] + s) #as the pfdf gets updated as well
#            #performance df
#            performance_df = insert_row(idx, performance_df, performance_df.iloc[idx,:])
#            #stereo matrix modification: move every element right and under the 1st double to the right
#            idx_rows_2nd = np.where(S_constraint[:,idx]!=0)[0][1] #identify
#            S_constraint[:,(idx+2):(S_constraint.shape[1]-len_stereo_doubles+s+1)] = S_constraint[:,(idx+1):(S_constraint.shape[1]-len_stereo_doubles+s)] #
#            S_constraint[idx_rows_2nd:S_constraint.shape[0],idx+1] =  S_constraint[idx_rows_2nd:S_constraint.shape[0],idx]
#            S_constraint[idx_rows_2nd:S_constraint.shape[0],idx]=0
#            
#        len_x = performance_df.shape[0]
        
    
    
    
    #which request can be reached by which satellites
    number_of_reach_areas_k = [[] for _ in range(len(location_slots))]
    string_list_np_k = [[] for _ in range(len(location_slots))]
    index_sat_k = [[] for _ in range(len(location_slots))]
    pfloc_np_k = [[] for _ in range(len(location_slots))]
    for k in range(0,len(location_slots)):
        which = performance_df["satellite"] == k
        reachable_k = np.unique(performance_df["request location"][which])
        if len(reachable_k) == 0:
            continue
        number_of_reach_areas_k[k].append(len(reachable_k))
        string_list_np_k[k].append(np.concatenate(reachable_k).reshape(((number_of_reach_areas_k[k][0]), 2)))
        index_sat_k[k].append(np.array(performance_df["request location"][performance_df["satellite"]==k].index))
        pfloc_np_k[k].append(np.concatenate(np.array(performance_df["request location"][performance_df["satellite"]==k])).reshape((len(index_sat_k[k][0]),2)))
        
    #### max 1 attempt per TIMESTEP per satellite
    A = np.zeros((number_of_acq_points, len_x, len(location_slots)))
    for k in range(0, len(location_slots)):
        if len(index_sat_k[k]) == 0:  #if satellite cannot reach any req
            continue
        time_sat_k = performance_df["time"][index_sat_k[k][0]]
        if len(time_sat_k) != 0:
            for i in range(0,len(time_sat_k)):
                time_index_i = np.where(np.array(time_slots, dtype = "datetime64[ns]") == np.array(time_sat_k)[i])[0][0]
                alt_index_old = time_sat_k.index[i]
                #alt_index_new = np.where(np.array(performance_df.index) == alt_index_old)[0][0]
                A[time_index_i,alt_index_old,k] = 1 
    
    #into LPP setup
    A_constraint = np.concatenate([A[:,:,k] for k in range(0,len(location_slots))], axis = 0)
        
            
    #### max h attempt per REQUEST
    #how many is inserted of the requests?
    string_list = list()
    BB = list(performance_df["request location"])
    string_list = list(set(list(map(str, BB))))

    number_of_reach_areas = len(string_list)
    import ast
    string_list_np = np.array([ast.literal_eval(n) for n in string_list])
    pfloc_np = np.array([xi for xi in performance_df["request location"]])
    B = np.zeros((number_of_reach_areas, len_x))
    B_rhs = np.ones((number_of_reach_areas))
    for i in range(0,number_of_reach_areas):
        B[i,:] = np.sum((pfloc_np == string_list_np[i,:]), axis = 1) == 2
        one_index_reach = np.where(np.sum(pfloc_np == string_list_np[i], axis = 1) == 2)[0][0]
        B_rhs[i] = max(performance_df["stereo"][one_index_reach]+1, performance_df["strips"][one_index_reach])
    
    #### only possible MANEUVERS           
    bar_i = bar_i + 1
    bar.update(bar_i)
    
                
    #maximum infeasibility range for a satellite:
    # rotational degree for satellite (horizon span dep on sat height)
    max_degree = 180-(2*np.degrees(np.arccos(6371/(6371+height_satellite))))
    T_man_acq_max = max_degree/rotation_speed + max(performance_df["duration"])
    t_step_search = int(np.floor(T_man_acq_max/seconds_gran))
    
    
    #alternativewise infeasible maneuvers - possible due to 1 acq per. req constraint
    FF = np.zeros((len_x, len_x))
    np.fill_diagonal(FF, 1)
    non_empty_sat_set = list()
    for k in range(len(location_slots)):
        if len(index_sat_k[k]) != 0:
            non_empty_sat_set.append(k)
    
    for i in range(0, len_x):
        #Only check those alternatives that are the same satellite
        i_is_in_sat = non_empty_sat_set[np.where([(i in index_sat_k[k][0]) for k in non_empty_sat_set])[0][0]]
        same_sat_alt = index_sat_k[i_is_in_sat]
        
        if len(same_sat_alt) == 0:
            continue
        
        t1 = np.where(A[:,i,i_is_in_sat])[0][0]
        #we move ahead to search those with a chance of not being feasible
        t2 = t1+t_step_search
        it = np.where(A[t1:t2,:,i_is_in_sat])[-1][-1]
        check_alternatives0 = list(range(i+1,it+1))
        
#        here we check for feasibility: among one attempt relative to all others
#        and this outcommented section removes all same request infeasibility and same time infeasibility (they were previously incorporated through other seperate constraints)        
#        #however we do not check alternatives that are the same request - maybe modify
#        #B row index of same request area 
#        b_index = np.where(np.sum(string_list_np == performance_df["request location"][i], axis = 1) == 2)[0][0]
#        same_req = np.where(B[b_index,:])[0]
#        
#        
#        #however we do not check alternatives that are the same request - maybe modify
#        #A row index of same time acquisition 
#        a_index = np.where(A[:, i, i_is_in_sat])[0]
#        same_time = np.where(A[a_index, :, i_is_in_sat])[0] 
#        
#        
#        
#        #remove - keep only the alternatives that are the same satellite 
#        #-  and which not attempt the same request
#        #-  and which not attempt from same location
#        rem_alt_sat = np.isin(np.array(check_alternatives0), same_sat_alt) 
#        rem_alt_req = np.isin(np.array(check_alternatives0), same_req, invert = True)                
#        rem_alt_time = np.isin(np.array(check_alternatives0), same_time)
#        #union
#        rem_alt = np.logical_and(rem_alt_req, rem_alt_sat, rem_alt_time)
#        
        rem_alt = np.isin(np.array(check_alternatives0), same_sat_alt)
        
        check_alternatives = list(np.array(check_alternatives0)[rem_alt])
        
        bar_i = bar_i + 1
        bar.update(bar_i)

        if (len(check_alternatives) == 0):
            continue
        
        #which request location is j's, split them up into different indexes
        unique_js = np.unique(pfloc_np[check_alternatives], axis=0)
        len_unique_js = len(unique_js)
        index_for_uniques = [np.where(np.sum(pfloc_np[check_alternatives] == np.array(i),axis=1)==2)[0] for i in list(unique_js)] 
            
        #calculate the vector for satellite to location i  
        sat_xyz1 = cart_system(performance_df["satellite location"][i][0],performance_df["satellite location"][i][1],height_satellite)
        loc_xyz1 = cart_system(performance_df["request location"][i][0],performance_df["request location"][i][1],0)
        vec1 = sat_xyz1-loc_xyz1
        
        #for each unique req, go from farthest away alternative, when it is infeasible, the rest is also infeasible
        for unique in range(0,len_unique_js):
            unique_req_js = list(np.array(check_alternatives)[index_for_uniques[unique]])
            unique_req_js.reverse()
            for j in range(0,len(unique_req_js)):
                #calculate the vector for satellite to location j     
                sat_xyz2 = cart_system(performance_df["satellite location"][unique_req_js[j]][0],performance_df["satellite location"][unique_req_js[j]][1],height_satellite)
                loc_xyz2 = cart_system(performance_df["request location"][unique_req_js[j]][0],performance_df["request location"][unique_req_js[j]][1],0)
                vec2 = sat_xyz2-loc_xyz2
                
                cos_radia = np.min([np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), 1]) #rounding error
                angle = np.degrees(np.arccos(cos_radia))
                
                #maneuvering i->j: seconds it takes to rotate satellite from diff. acquisitions
                T_man_i2j = angle/rotation_speed 
                
                if (T_man_i2j + performance_df.iloc[i]["duration"] > (performance_df.iloc[unique_req_js[j]]["time"] - performance_df.iloc[i]["time"]).total_seconds()):
                    FF[i,unique_req_js[j:]] = 1
                    
    #remove those with only it self as a request
    if simplify == True:    
        #keep = np.where(np.sum(FF, axis = 1)!=1)[0]
        #FF = FF[keep, :]
        F = FF
        
    A_constraint = A_constraint[np.sum(A_constraint, axis = 1)!=0,:] #time
    
    if simplify == False:
        #find complete infeasible sets in FF 
        FF_new = np.array(FF)
        infeasible_sets = np.zeros((1,FF.shape[1]))
        i=0
        while i < FF.shape[0]:
            i_infeasible_pairwise = np.where(FF_new[i,:])[0]
            if len(i_infeasible_pairwise) <= 2:
                i=i+1
            else:
                cols_modified = np.where(np.sum(FF[i_infeasible_pairwise,:][:,i_infeasible_pairwise], axis = 0) - np.arange(1,len(i_infeasible_pairwise)+1) == 0)[0]
                if len(cols_modified) < 2:
                    i=i+1
                else:
                    complete_infeasible_set = i_infeasible_pairwise[cols_modified]
                    complete_infeasible_set_array = np.zeros((1,FF.shape[1])) 
                    complete_infeasible_set_array[0,complete_infeasible_set] = 1
                    
                    included_com_set = np.any(np.sum(infeasible_sets - complete_infeasible_set_array <0, axis = 1) == 0)
                    if included_com_set: #if already included move on
                        i=i+1 
                    else: #else include it in the infeasible sets
                        infeasible_sets = np.concatenate((infeasible_sets, complete_infeasible_set_array), axis = 0)
        
        #create F_complete set
        F_set = np.delete(infeasible_sets, 0, axis = 0)
        #remove these from FF_new
        for i in range(1,infeasible_sets.shape[0]):
            complete_index = np.where(infeasible_sets[i,:])[0]
            FF_new[complete_index,complete_index] = 0
                
        #pairwise infeasible maneuvers
        np.fill_diagonal(FF_new,0) 
        total_FF = int(np.sum(FF_new))
        F_pair = np.zeros((total_FF, FF_new.shape[1]))
        FF_index = np.arange(total_FF)
        FF_pair = np.where(FF_new == 1)
        F_pair[FF_index, FF_pair[0]] = 1 
        F_pair[FF_index, FF_pair[1]] = 1 
        
        ###check if pair is in A or B
        f1 = F_pair.shape[0]
        for i in range(0, F_pair.shape[0]):
            in_A = np.any(np.sum(A_constraint- F_pair[i,:] <0, axis = 1) == 0)
            in_B = np.any(np.sum(B - F_pair[i,:] <0, axis = 1) == 0)
            if in_A or in_B:
                F_pair[i,:] = 0
        F_pair = F_pair[np.sum(F_pair, axis = 1)!=0,:] #time
        print("for F_pair in A,B - dropped:", f1,  'to', F_pair.shape[0], "constraints")
    
    
        #concatenate into one F
        F = np.concatenate((F_pair,F_set), axis = 0)
        
        
        ####check if a or B is in F_set
        len_ab = A_constraint.shape[0]+B.shape[0]
        for i in range(0, A_constraint.shape[0]):
            in_f = np.any(np.sum(F_set - A_constraint[i,:] <0, axis = 1) == 0)
            if in_f:
                A_constraint[i,:] = 0
        A_constraint = A_constraint[np.sum(A_constraint, axis = 1)!=0,:] #time
        for i in range(0, B.shape[0]):
            in_f = np.any(np.sum(F_set - B[i,:] <0, axis = 1) == 0)
            if in_f:
                B[i,:] = 0
                B_rhs[i] = 0 
        B = B[np.sum(B, axis = 1)!=0,:] #time
        B_rhs = B_rhs[B_rhs != 0] #r h s
        print("for A,B in F_Set - dropped:", len_ab,  'to', A_constraint.shape[0]+B.shape[0], "constraints")   
        
    
    
    bar_i = bar_i + 1
    bar.update(bar_i)
    

    ## CAPACITY constraint
    compression_factor = 2
    K = (performance_df["area"]*(1/cam_resolution))/compression_factor
    K = np.array(K).reshape((1,len(K)))
    #L = np.array(capacity_limit).reshape((1,1))
        

    #STRIPS constraint
    strips_perf_df = np.where((performance_df["stereo"]<2)[performance_df["strips"]>=2])[0]
    strips_IDs = list(np.unique(performance_df["ID"][strips_perf_df]))
    number_of_strips_IDs = list()
    Strips = list()
    for i in range(0,len(strips_IDs)):
        range_strip = list(np.where(performance_df["ID"] == strips_IDs[i])[0])
        strips_for_i = performance_df["strips"][range_strip[0]]
        if len(range_strip) < strips_for_i:
            continue
        for a in range(0,int(len(range_strip)-strips_for_i)):
            Strips.append(range_strip[a:])
            number_of_strips_IDs.append(strips_for_i)
    
    Strips_constraint = np.zeros((len(Strips),len_x))
    for i in range(len(Strips)):
        Strips_constraint[i,Strips[i][0]] = number_of_strips_IDs[i]
        Strips_constraint[i,Strips[i][1:]] = -1
        
    
    #### OPTIMIZATION
    #less than constraint
    #LHS
    A_constraint.shape
    B.shape
    F.shape
    LESS_THAN_Matrix = np.concatenate((B,A_constraint,F), axis = 0) #(A_constraint)
    LESS_THAN_Matrix.shape
    
    #RHS
    #one acq per request except for stereo -> it is there
    B_rhs.shape
    F_rhs = np.ones((F.shape[0]+A_constraint.shape[0],1))  #added A
    F.shape
    rhs_ABF = np.concatenate((B_rhs,np.squeeze(F_rhs)), axis = 0)
    rhs_ABF.shape
    #rhs_ABFL = np.concatenate((rhs_ABF, L), axis = 0)
    #rhs_ABFL.shape
    
    #equal to constraint
    S_constraint.shape
    Strips_constraint.shape
    eLHS = S_constraint #np.concatenate((S_constraint, Strips_constraint), axis = 0)   
    eRHS = np.zeros(eLHS.shape[0])
    
    #drop empty rows
    non_empty_rows = ~(np.sum(LESS_THAN_Matrix, axis = 1)==0)
    LHS_leq = LESS_THAN_Matrix[non_empty_rows,:]
    RHS_leq = rhs_ABF[non_empty_rows]
    RHS_leq.shape
    
    #convert LHS matrix to sparce matrix
    sparcity = np.sum(LHS_leq == 0)/(LHS_leq.shape[0]*LHS_leq.shape[1])  #level of sparsity
    if sparcity >= 0.75:
        from scipy import sparse
        b=sparse.csr_matrix(LHS_leq)
        from scipy_sparce_to_spmatrix import scipy_sparse_to_spmatrix
        LHS_leq = scipy_sparse_to_spmatrix(b)
    if sparcity < 0.75:
        from cvxopt import matrix
        LHS_leq = matrix(LHS_leq)
    #################### LPP DONE ################################
    
    
    bar.finish()

    LPP_data.B = B
    LPP_data.stereo = S_constraint
    LPP_data.strips = Strips_constraint
    LPP_data.performance_df = performance_df
    LPP_data.F = FF
    LPP_data.LHS = LHS_leq
    LPP_data.RHS = RHS_leq
    LPP_data.eLHS = eLHS
    LPP_data.eRHS = eRHS 
    return(LPP_data)
 














############### LPP data computation single satellite
def LPP_data(performance_df, number_of_acq_points, time_slots, location_slots, 
             height_satellite, rotation_speed, seconds_gran,
             sat_energy_0, capacity_limit, cam_resolution):
    
    #packages
    import ephem
    import numpy as np
    import progressbar

    bar = progressbar.ProgressBar(max_value=performance_df.shape[0]+4)
    bar_i = 1
    bar.update(bar_i)
    
    
    
    
    len_x = performance_df.shape[0]
    #### max 1 attempt per TIMESTEP
    A = np.zeros((number_of_acq_points, len_x))
    for i in range(0,number_of_acq_points):
        A[i,:] = performance_df["time"] == time_slots[i]
            

    #### max 1 attempt per REQUEST
    #how many is inserted of the requests?
    string_list = list()
    BB = list(performance_df["request location"])
    string_list = list(set(list(map(str, BB))))

    number_of_reach_areas = len(string_list)
    import ast
    string_list_np = np.array([ast.literal_eval(n) for n in string_list])
    pfloc_np = np.array([xi for xi in performance_df["request location"]])
    B = np.zeros((number_of_reach_areas, len_x))
    for i in range(0,number_of_reach_areas):
        B[i,:] = np.sum((pfloc_np == string_list_np[i,:]), axis = 1) == 2
    
    
    #### only possible MANEUVERS
                
    #conversion into cartesian coordinate system. Note R is appr. radius of earth 6371 km
    def cart_system(lat, lon, elevation):
        import numpy as np 
        lat = np.radians(lat)
        lon = np.radians(lon)
        R = 6371 + elevation
        x = R * np.cos(lat) * np.cos(lon)
        y = R * np.cos(lat) * np.sin(lon)
        z = R * np.sin(lat)
        return(np.array([x,y,z]))

    
    #requests with chance of infeasibility maneuvers - most extreme angular rotation tests for each pair of request
    infeasibel_pair_extreme_requests = np.zeros((number_of_reach_areas, number_of_reach_areas))
    for i in range(0,number_of_reach_areas):
        for j in range(0,number_of_reach_areas):
            if i == j:
                infeasibel_pair_extreme_requests[i,j] = 1
            else:
                arg_max_i = np.argmax(performance_df["time"][np.sum(pfloc_np == string_list_np[i], axis = 1) == 2])            
                time_min_i = np.min(performance_df["time"][np.sum(pfloc_np == string_list_np[i], axis = 1) == 2])
                #time_max_i = performance_df["time"][arg_max_i]
                
                arg_min_j = np.argmin(performance_df["time"][np.sum(pfloc_np == string_list_np[j], axis = 1) == 2])
                #time_min_j = performance_df["time"][arg_min_j]
                time_max_j = np.max(performance_df["time"][np.sum(pfloc_np == string_list_np[j], axis = 1) == 2]) 
                
                if (time_max_j < time_min_i):
                    infeasibel_pair_extreme_requests[i,j] = 1
                else:
                    sat_xyz_i = cart_system(performance_df["satellite location"][arg_max_i][0],performance_df["satellite location"][arg_max_i][1],height_satellite)
                    loc_xyz_i = cart_system(performance_df["request location"][arg_max_i][0],performance_df["request location"][arg_max_i][1],0)
                    vec_i = sat_xyz_i-loc_xyz_i
                        
                    sat_xyz_j = cart_system(performance_df["satellite location"][arg_min_j][0],performance_df["satellite location"][arg_min_j][1],height_satellite)
                    loc_xyz_j = cart_system(performance_df["request location"][arg_min_j][0],performance_df["request location"][arg_min_j][1],0)
                    vec_j = sat_xyz_j-loc_xyz_j
                        
                    angle = np.arccos(np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j)))
                    #maneuvering i->j: seconds it takes to rotate satellite from diff. acquisitions
                    T_man_i2j = np.degrees(angle)/rotation_speed 
                    
                    if (T_man_i2j + performance_df.iloc[arg_max_i]["duration"] > (performance_df.iloc[arg_min_j]["time"] - performance_df.iloc[arg_max_i]["time"]).total_seconds()):
                        infeasibel_pair_extreme_requests[i,j] = 2

    bar_i = bar_i + 1
    bar.update(bar_i)

                
    #maximum infeasibility range:
    # rotational degree for satellite (horizon span dep on sat height)
    max_degree = 180-(2*np.degrees(np.arccos(6371/(6371+height_satellite))))
    T_man_acq_max = max_degree/rotation_speed + max(performance_df["duration"])
    t_step_search = int(np.floor(T_man_acq_max/seconds_gran))
    
    
    #alternativewise infeasible maneuvers - possible due to 1 acq per. req constraint
    F = np.zeros((len_x, len_x))
    np.fill_diagonal(F, 1)
    for i in range(0, len_x):
        #is there a request where there is a chance for infeasible maneuvers?
        which_req = np.where(np.sum(string_list_np == pfloc_np[i], axis = 1)==2)[0][0]
        any_man_calc = any(infeasibel_pair_extreme_requests[which_req,:]==2)
        if (any_man_calc):
            t1 = np.where(A[:,i])[0][0]
            #we move ahead to search those with a chance of not being feasible
            t2 = t1+t_step_search
            it = np.where(A[t1:t2,:])[-1][-1]
            check_alternatives0 = list(range(i,it+1))
            #however we do not check alternatives that are the same request - maybe modify
            #B row index of same request area 
            b_index = np.where(np.sum(string_list_np == performance_df["request location"][i], axis = 1) == 2)[0][0]
            same_alt = np.where(B[b_index,:])[0]
        
            #remove
            rem_alt = np.isin(np.array(check_alternatives0), same_alt, invert = True)
            check_alternatives = list(np.array(check_alternatives0)[rem_alt])
            
            bar_i = bar_i + 1
            bar.update(bar_i)

            
            if (len(check_alternatives) == 0):
                continue
        
            #which request location is j's, split them up into different indexes
            unique_js = np.unique(pfloc_np[check_alternatives], axis=0)
            len_unique_js = len(unique_js)
            index_for_uniques = [np.where(np.sum(pfloc_np[check_alternatives] == np.array(i),axis=1)==2)[0] for i in list(unique_js)] 
            
            #calculate the vector for satellite to location i  
            sat_xyz1 = cart_system(performance_df["satellite location"][i][0],performance_df["satellite location"][i][1],height_satellite)
            loc_xyz1 = cart_system(performance_df["request location"][i][0],performance_df["request location"][i][1],0)
            vec1 = sat_xyz1-loc_xyz1
            
            #for each unique req, go from farthest away alternative, when it is infeasible, the rest is also infeasible
            for unique in range(0,len_unique_js):
                unique_req_js = list(np.array(check_alternatives)[index_for_uniques[unique]])
                unique_req_js.reverse()
                for j in range(0,len(unique_req_js)):
                    #calculate the vector for satellite to location j     
                    sat_xyz2 = cart_system(performance_df["satellite location"][unique_req_js[j]][0],performance_df["satellite location"][unique_req_js[j]][1],height_satellite)
                    loc_xyz2 = cart_system(performance_df["request location"][unique_req_js[j]][0],performance_df["request location"][unique_req_js[j]][1],0)
                    vec2 = sat_xyz2-loc_xyz2
                    
                    angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
                    #maneuvering i->j: seconds it takes to rotate satellite from diff. acquisitions
                    T_man_i2j = np.degrees(angle)/rotation_speed 
                    
                    if (T_man_i2j + performance_df.iloc[i]["duration"] > (performance_df.iloc[j]["time"] - performance_df.iloc[i]["time"]).total_seconds()):
                        F[i,unique_req_js[j:]] = 1
            
    #remove those with only it self as a request
    keep = np.where(np.sum(F, axis = 1)!=1)[0]
    F = F[keep, :]

    ##pairwise infeasible maneuvers
    ##FF = np.array([[0,1,0,1],[0,0,1,1],[0,0,0,0],[0,0,0,0]])
    #F = np.zeros((int(np.sum(FF)), FF.shape[1]))
    #index_list = [0] + list(map(int, np.sum(FF, axis = 1)))
    #cumsum_list = np.cumsum(index_list)
    #for i in range(0,len(index_list)-1):
    #    index_i = list(range(cumsum_list[i], cumsum_list[i+1])) 
    #    if (len(index_i) != 0):
    #        F[index_i,i] = 1
    #        j_index = np.where(FF[i,:]==1)[0]
    #        for j in range(0,len(j_index)):
    #            F[index_i[j],j_index[j]] = 1


    #### BATTERY CONSTRAINT
    #energy cost
    #note it is dependent on area, shape dificulty
    #cost in pct function: distance --> degrees
    #cost = degrees * pct/degree + strips * pct/strip
    #               cost_rotation           cost_strip
    cost_rotation = 0.20    #pct/degree of rotation
    #cost_strip = 1          #pct/strip
    cost_matrix = np.zeros((number_of_acq_points, len_x))
    len_indices = 0
    cost_indices = 0
    for i in range(0,number_of_acq_points):
        indices = np.where(performance_df["time"] <= time_slots[i])[0]
        if len(indices)-len_indices != 0:
            len_indices = len(indices)
            cost_indices = performance_df["angle"][indices] * cost_rotation + performance_df["complexity"][indices]
            cost_matrix[i,indices] = cost_indices
        else:
            cost_matrix[i,indices] = cost_indices


    #cumsum operator
    cumsum_operator = np.zeros((number_of_acq_points, number_of_acq_points))
    for i in range(0,number_of_acq_points):
        cumsum_operator[i,:(i+1)] = np.ones(i+1)

    #energy start level
    energy_start_vector = np.zeros((number_of_acq_points, 1))
    energy_start_vector[:,0] = sat_energy_0

    #energy gain
    #note it is dependent on sun exposure/elevation relative TO SATELLITE! 

    #sun elevation to satellite 
    R_earth = 6371
    add_horizon_angle = np.degrees(np.arccos(R_earth/(R_earth+height_satellite)))
    sun_elevation_satellite = list()
    for i in range(0,number_of_acq_points):
        #get sun position for earth location
        obs = ephem.Observer()
        obs.lat = str(location_slots[i][0])
        obs.long = str(location_slots[i][1])
        obs.date = time_slots[i]
        sun = ephem.Sun(obs)
        sun.compute(obs)
        sun_angle = np.degrees(sun.alt) # Convert Radians to degrees
        #satellite sun elevation over horizon
        sun_elevation_satellite.append(sun_angle + add_horizon_angle)

    ###energy gain modelled
    #max energy increase per minut in pct 
    max_energy_increase = 1
    #normalized to fit timestep discretization
    max_energy = (max_energy_increase/60)*seconds_gran
    #angular framework change to fit sinusoid behavior of energy gain
    #i.e. max angel: 90+theta in old framework -> 90 in new framework 
    angular_change_constant = 90/(90+add_horizon_angle)
    gain_matrix = np.zeros((number_of_acq_points, 1))
    for i in range(0,number_of_acq_points):
        sun_horizon_angel_i = sun_elevation_satellite[i]
        if sun_horizon_angel_i < 0:
            sun_horizon_angel_i = 0
        gain_matrix[i] = np.sin(np.radians(angular_change_constant * sun_horizon_angel_i)) * max_energy


    #   #linearize constraint so loss will never be more than 100 pct
    #   #LHS
    #   alt_timesteps = np.unique(np.array(performance_df["time"]))
    #   I_len = sum(range(0,len(alt_timesteps)))
    #   I = np.zeros((I_len,len_x))
    #   i=0
    #   for m in range(0,len(alt_timesteps)):
    #       for n in range(m,len(alt_timesteps)-1):
    #           for a in range(0,len(list(range(m,n+2)))):
    #               which_a = np.array(performance_df["time"]) == alt_timesteps[list(range(m,n+2))][a]
    #               I[i,which_a] = 1
    #           i=i+1
    #   #multiply elementwise the battery cost for each particular maneuver 
    #   I_c = np.multiply(I,np.reshape(list(cost_matrix[-1,:])*I_len,(I_len,len_x)))
    #   
    #   #RHS
    #   R_len = sum(range(0,len(alt_timesteps)))
    #   R = np.zeros((I_len,number_of_acq_points))
    #   i=0
    #   for m in range(0,len(alt_timesteps)):
    #       for n in range(m,len(alt_timesteps)-1):
    #           which_b = ((np.array(time_slots, dtype = "datetime64") >= alt_timesteps[list(range(m,n+2))][0]) & (np.array(time_slots, dtype = "datetime64") <= alt_timesteps[list(range(m,n+2))][-1]))
    #           R[i,which_b] = 1
    #           i=i+1
    #   #   #gain in the respective intervals + 100
    #   I_g100 = R@gain_matrix + 100
    
    bar_i = bar_i + 1
    bar.update(bar_i)

    
    ####energy constraint setup  #overall use<=tank and interval use<interval gain + maxdrop allowed (100)
    #C = np.concatenate((cost_matrix, I_c), axis = 0)
    C = cost_matrix
    E1 = energy_start_vector + cumsum_operator @ gain_matrix
    #E = np.concatenate((E1, I_g100), axis = 0)
    E = E1


    ## CAPACITY constraint
    compression_factor = 2
    K = (performance_df["area"]*(1/cam_resolution))/compression_factor
    K = np.array(K).reshape((1,len(K)))
    L = np.array(capacity_limit).reshape((1,1))
    
    #### OPTIMIZATION
    #less than constraint
    #LHS
    A.shape
    B.shape
    F.shape
    C.shape
    LESS_THAN_Matrix = np.concatenate((A,B,F,C,K), axis = 0)
    LESS_THAN_Matrix.shape
    
    #RHS
    rhs_ABF = np.ones((A.shape[0]+B.shape[0]+F.shape[0],1))
    rhs_ABF.shape
    E.shape
    rhs_ABFCEL = np.concatenate((rhs_ABF, E, L), axis = 0)
    rhs_ABFCEL.shape
    
    
    #drop empty rows
    non_empty_rows = ~(np.sum(LESS_THAN_Matrix, axis = 1)==0)
    LHS_leq = LESS_THAN_Matrix[non_empty_rows,:]
    RHS_leq = rhs_ABFCEL[non_empty_rows,:]
    RHS_leq.shape
    
    #convert LHS matrix to sparce matrix
    sparcity = np.sum(LHS_leq == 0)/(LHS_leq.shape[0]*LHS_leq.shape[1])  #level of sparsity
    if sparcity >= 0.75:
        from scipy import sparse
        b=sparse.csr_matrix(LHS_leq)
        from scipy_sparce_to_spmatrix import scipy_sparse_to_spmatrix
        LHS_leq = scipy_sparse_to_spmatrix(b)
    if sparcity < 0.75:
        from cvxopt import matrix
        LHS_leq = matrix(LHS_leq)
    #################### LPP DONE ################################
    
    
    bar.finish()

    
    LPP_data.LHS = LHS_leq
    LPP_data.RHS = RHS_leq
    LPP_data.battery_C = C
    LPP_data.battery_E = E
    return(LPP_data)
   