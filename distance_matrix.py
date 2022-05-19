#distance matrix

def distance_matrix(location_slots, df, max_off_nadir_angle, height_satellite, number_of_satellites):
    from math import floor
    from geopy.distance import great_circle
    import numpy as np
    
    #calculate reachability based on max_off_nadir_angle
    #note fat angled triangle can occur
    
    R_earth = 6371
    phi = 180 - np.degrees(np.arcsin(((R_earth+height_satellite)/R_earth)*np.sin(np.radians(max_off_nadir_angle))))
    yotta = np.radians(180-max_off_nadir_angle-phi)
    reachability = R_earth * yotta
    d_horizon = R_earth * np.arccos(R_earth/(R_earth*height_satellite))
    if reachability > d_horizon or np.isnan(reachability):
        reachability = d_horizon
                
    #relies on the triangle inequality - does not have to check all distances
    #if distance from sat_i to req is 3 times the reachability, the next 2 sat
    #locations are excluded of calculation - naturally not reachable
    len_loc = len(location_slots[0])
    distance = np.zeros((len_loc, df.shape[0], number_of_satellites))
    step_distance = great_circle((location_slots[0][0][0],location_slots[0][0][1]), (location_slots[0][1][0],location_slots[0][1][1])).kilometers    
    for k in range(0,number_of_satellites):
        for i in range(0,df.shape[0]):                      #req position
            n=0
            while (n < len_loc):                #sat position 
                                                 #list                                        #panda
                dist = great_circle((location_slots[k][n][0],location_slots[k][n][1]), (df.iloc[i]["request location"][0], df.iloc[i]["request location"][1])).kilometers
                if dist>reachability:
                    step_checker = max(floor((dist-reachability)/step_distance),1)
                    n_2 = min(n+step_checker, len_loc)
                    interval = list(range(n,n_2))
                    distance[interval,i,k] = np.nan
                    n = n + step_checker
                else: 
                    distance[n,i,k] = dist
                    n = n + 1
    return(distance)
    
