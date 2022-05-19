def parallelectre(dat, q, p, v, objective_data, weights):
    import numpy as np
    import timeit
    start = timeit.default_timer()
        
    #normalise data, create empty dataframe
    data_WN = np.copy(dat) #np.zeros((data.shape[0],data.shape[1]))
    data_shape = data_WN.shape
    
    #objectify data
    for i in range(0,data_shape[0]):
        if objective_data[i] == 0:
            data_WN[i,:] = np.max(data_WN[i,:])-data_WN[i,:]    
        
    #calculate concordance 
    CM_temp = np.zeros((data_shape[1],data_shape[1]))
    S_temp = np.zeros((data_shape[1],data_shape[1]))
    qq = np.array(data_shape[1]*list(q)).reshape(data_shape[1], data_shape[0]).T
    pp = np.array(data_shape[1]*list(p)).reshape(data_shape[1], data_shape[0]).T
    vv = np.array(data_shape[1]*list(v)).reshape(data_shape[1], data_shape[0]).T
    
    for i in range(0, data_shape[1]):                     #i is a, j is all other
        #for j in range(0, data.shape[1]):
        #if i==j:
        #CM_temp[i,i] = 1
        #S_temp[i,i] = 1
        #else:
        diff_vector = data_WN - np.array(data_shape[1]*list(data_WN[:,i])).reshape(data_shape[1], data_shape[0]).T
        #concordance if
        phi = np.zeros(diff_vector.shape)
        phi[diff_vector <= qq] = 1 
        a = (qq < diff_vector) & (diff_vector < pp)
        phi[a] = (pp[a] - diff_vector[a])/(pp[a]-qq[a])
        #phi[pp <= diff_vector] = 0          #these are already denoted 0
        #discordance if
        d = np.zeros(diff_vector.shape)
        d[diff_vector >= vv] = 1
        b = (pp < diff_vector) & (diff_vector < vv)
        d[b] = (diff_vector[b]-pp[b])/(vv[b]-pp[b])
        #d[diff_vector<=p] = 0              #these are already denoted 0
        
        #overall concordance and credibility index
        CM_temp[i,:] = np.dot(weights, phi)/np.sum(weights)
        CC = np.array(data_shape[0]*list(CM_temp[i,:])).reshape(data_shape[0], data_shape[1])
        
        all_true = np.sum(d<=CC, axis = 0) == data_shape[1]
        S_temp[i, all_true] = np.array(CM_temp[i,all_true])
        
        K = ((d>CC) & (CC != 1))[:,~all_true]
        prod_m = ~K*1
        #upper
        prod_m_upper = np.array(prod_m, dtype = "float64")
        #np.put(prod_m_upper, np.squeeze(np.where(K.reshape((K.shape[0]*K.shape[1]))))[np.squeeze(np.where(~all_true))], (1-d[K])[~all_true])
        np.put(prod_m_upper, [np.where(K.reshape((K.shape[0]*K.shape[1])))], (1-d[:,~all_true][K]))
        upper = np.prod(prod_m_upper.reshape(K.shape), axis = 0)
        #lower
        #prod_m_lower = np.array(prod_m, dtype = "float64")
        #np.put(prod_m_lower, [np.where(K.reshape((K.shape[0]*K.shape[1])))], (1-CM_temp[i,~all_true]))
        #lower = np.prod(prod_m_lower.reshape(K.shape), axis = 0)
        
        lower = (1-CM_temp[i,~all_true])**(np.sum(K, axis =0))
                  
        S_temp[i,~all_true] = CM_temp[i,~all_true] * (upper/lower)
        
    #print(np.array_str(CM_temp, precision = 2, suppress_small=True))
    #print(np.array_str(S_temp,  precision = 2, suppress_small=True))
    

    
    end = timeit.default_timer()
    time_electre = end - start
    
    #electreiii.description = "Of a to b, 3 means prefered, 2 means indifferent, 1 means incompatible, and 0 means not prefered over. Remember that for prelimenary rankings that Python counts from 0"
    parallelectre.time = time_electre
    parallelectre.score = S_temp
    #electreiii.top = D_distil_rank
    #electreiii.bottom = A_distil_rank
    #electreiii.final = final_ranking
    return(parallelectre)