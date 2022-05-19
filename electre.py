#ELECTRE III
#import numpy as np

#     alt1 , alt2 , . . . .
#cri1
#cri2
#  .
#  .                       meaning alternative1 in coloumn1 and its criterias in each row

#data = np.array([[25,10],[20,30],[15,20]])

#HERE the DM must set some thresholds:

#Preference threshold [pi]: is a difference above which the decision maker strongly
#prefers a management alternative over all for the criterion i.
#Alternative b is strictly preferred to alternative a in terms of criterion i if
#gi(b) > gi(a) + p(gi(a)). 

#Indifference threshold [qi]: is a difference beneath which the decision maker is
#indifferent between two management alternatives for the criterion i.
#Alternative b is weakly preferred to alternative a in terms of criterion i if
#gi(b) > gi(a) + q(gi(a)). 

#Veto threshold [vi]: blocks the outranking relationship between alternatives 
#for the criterion i. Alternative a cannot outrank alternative b if the 
#performance of b exceeds that of a by an amount greater than the veto 
#threshold, i.e. if gi(b) ≥ gi(a) + vi(gi(a)). 

#steps:
#1) The construction of a valued outranking relation;
#2) The construction of two complete preorders based on descending and
#ascending distillation chains;
#3) The comparison of the two complete preorders in order to elaborate a
#final ranking of the alternatives. This comparison leads to a partial
#preorder in which it is possible that some alternatives are incomparable.

#Algorithm:

#The start point is the decision matrix. The parameters pi, qi and vi have to be defined by
#the user. 

#setup data, and the parameters set by the DM (note for all criterias i, and
#the thresholds are here not dependent on the performance of the alternatives):
#p = np.random.randint(5, 10, data.shape[0])/100
#q = np.random.randint(1, 5, data.shape[0])/100
#v = np.random.randint(10, 20, data.shape[0])/100
#percentaged_thresholds = True

def electreiii(data, p, q, v, objective_data, weight_criteria):
    import numpy as np
    import timeit
    start = timeit.default_timer()
    #weights
    if (weight_criteria == None).all():
        weight_criteria = [1]*data.shape[0]
        
    #normalise data, create empty dataframe
    data_WN = data #np.zeros((data.shape[0],data.shape[1]))
        
    #normalize data, possibly thresholds, and weights
    #p = np.array(p, dtype = float)
    #q = np.array(q, dtype = float)
    #v = np.array(v, dtype = float)
    
    #for n in range(0, data.shape[0]):
    #    norm = float(np.sum(np.square(data[n,:]))**0.5)
    #    if norm == 0:
    #       norm = 1.0
    #    data_WN[n,:] = (np.array(data[n,:])/norm)    #sometimes norm=0
    #    if (percentaged_thresholds == False):
    #            p[n] = p[n]/norm
    #            q[n] = q[n]/norm
    #            v[n] = v[n]/norm
    
    #objectify data
    for i in range(0,data_WN.shape[0]):
        if objective_data[i] == 0:
            data_WN[i,:] = np.max(data_WN[i,:])-data_WN[i,:]    
        
    #weight_criteria = np.array(weight_criteria)/sum(weight_criteria)
    
    #calculate concordance 
    CM_temp = np.zeros((data.shape[1],data.shape[1]))
    S_temp = np.zeros((data.shape[1],data.shape[1]))
    for i in range(0, data.shape[1]):
        for j in range(0, data.shape[1]):
            if i==j:
                CM_temp[i,j] = 1
                S_temp[i,j] = 1
            else:
                diff_vector = data_WN[:,j] - data_WN[:,i]
                #concordance if
                phi = np.zeros(diff_vector.shape[0])
                phi[diff_vector <= q] = 1 
                a = (q < diff_vector) & (diff_vector < p)
                phi[a] = (p[a] - diff_vector[a])/(p[a]-q[a])
                phi[p <= diff_vector] = 0
                #discordance if
                d = np.zeros(diff_vector.shape[0])
                d[diff_vector >= v] = 1
                b = (p < diff_vector) & (diff_vector < v)
                d[b] = (diff_vector[b]-p[b])/(v[b]-p[b])
                d[diff_vector<=p] = 0
                
                #overall concordance and credibility index
                CM_temp[i,j] = np.dot(weight_criteria, phi)/np.sum(weight_criteria)
                if (all(d<=CM_temp[i,j]) | all(v == None)):
                    S_temp[i,j] = np.squeeze(CM_temp[i,j])
                else:
                    K = d>CM_temp[i,j]
                    S_temp[i,j] = CM_temp[i,j] * np.prod((1-d[K])/(1-CM_temp[i,j]))
    
    #print(np.array_str(CM_temp, precision = 2, suppress_small=True))
    #print(np.array_str(S_temp,  precision = 2, suppress_small=True))
    
# =============================================================================
#     #prelimenary ranking
#     #descending destillation koefficients 
#     alfa = -0.15
#     beta = 0.3
#     #descending destillation
#     D_distil_rank = list()
#     alter_left = list(range(0,data.shape[1]))
#     
#     lampda = S_temp[alter_left,:][:,alter_left].max()
#     Di = 0
#     while((len(alter_left) != 0)):
#         if len(alter_left) == 1:
#             D_distil_rank.append(alter_left[0])
#             break
#         if len(alter_left) < 1:
#             break
#         
#         lampda = lampda - (beta + alfa * lampda)
#         T = S_temp[alter_left,:][:,alter_left]>lampda
#         lampda_strength = np.sum(T, axis = 1)
#         lampda_weakness = np.sum(T.transpose(), axis = 1)
#         #lampda_strength = np.sum(S_temp[alter_left,:][:,alter_left]>lampda, axis=1)
#         #lampda_weakness = np.sum(((1-(beta+alfa*lampda))*S_temp[alter_left,:][:,alter_left])<S_temp[alter_left,:][:,alter_left].transpose(), axis = 1)
#         qualification = lampda_strength - lampda_weakness 
#         
#         Di = sum(qualification == max(qualification))
#         if (Di < 1):
#             continue
#         
#         rank_i = np.where(qualification == max(qualification))[0]
#         rank_true = list()
#         for i in range(0,len(rank_i)):
#             counter = alter_left[rank_i[i]]
#             rank_true.append(int(counter))
#         for i in range(0,len(rank_true)):
#             alter_left.remove(rank_true[i])
#         D_distil_rank.append(rank_true)
#          
#        
#     #ascending destillation koefficients 
#     alfa = -0.15
#     beta = 0.3
#     #Ascending destillation
#     A_distil_rank = list()
#     alter_left = list(range(0,data.shape[1]))
#     
#     lampda = S_temp[alter_left,:][:,alter_left].max()
#     Di = 0
#     while((len(alter_left) != 0) & (lampda > 0)):
#         if len(alter_left) == 1:
#             A_distil_rank.append(alter_left[0])
#             break
#         if len(alter_left) < 1:
#             break
#             
#         lampda = lampda - (beta + alfa * lampda)
#         T = S_temp[alter_left,:][:,alter_left]>lampda
#         lampda_strength = np.sum(T, axis = 1)
#         lampda_weakness = np.sum(T.transpose(), axis = 1)
#         #lampda_strength = np.sum(S_temp[alter_left,:][:,alter_left]>lampda, axis=1)
#         #lampda_weakness = np.sum(((1-(beta+alfa*lampda))*S_temp[alter_left,:][:,alter_left])<S_temp[alter_left,:][:,alter_left].transpose(), axis = 1)
#         qualification = lampda_strength - lampda_weakness 
#         
#         Di = sum(qualification == min(qualification))
#         if (Di < 1):
#             continue
#         
#         rank_i = np.where(qualification == min(qualification))[0]
#         rank_true = list()
#         for i in range(0,len(rank_i)):
#             counter = alter_left[rank_i[i]]
#             rank_true.append(int(counter))
#         for i in range(0,len(rank_true)):
#             alter_left.remove(rank_true[i])
#         A_distil_rank.append(rank_true)
# 
#     
#     #final ranking
#     def class_search(rank_list, x, y, reverse):
#         k = 0
#         for i in range(0,len(rank_list)):
#             if (len(rank_list) == 1):
#                 x_i = 0
#                 y_i = 0
#             else:
#                 try:
#                     if any(np.array(rank_list[i]) == x):
#                         x_i = i
#                         k = k+1
#                         if k==2:
#                             break
#                 except:
#                     if (np.array(rank_list[i]) == x):
#                         x_i = i
#                         k = k+1
#                         if k==2:
#                             break
#                 try:
#                     if any(np.array(rank_list[i]) == y):
#                         y_i = i
#                         k=k+1
#                         if k==2:
#                             break
#                 except:
#                     if (np.array(rank_list[i]) == y):
#                         y_i = i
#                         k=k+1
#                         if k==2:
#                             break
#             
#         if x_i < y_i:
#             if(reverse == False):
#                 return(3)
#             else:
#                 return(0)
#         if x_i > y_i:
#             if(reverse == False):
#                 return(0)
#             else:
#                 return(3)
#         if x_i == y_i:
#             return(2)
#     
#     final_ranking = np.zeros((data.shape[1],data.shape[1]))
#     for a in range(0,data.shape[1]):
#         for b in range(0,data.shape[1]):
#             if a==b:
#                 final_ranking[a,b] = 2
#             else:
#                 top = class_search(D_distil_rank, a, b, reverse = False)
#                 bot = class_search(A_distil_rank, a, b, reverse = True)
#                 if (top == bot):                                         # they then agree
#                     final_ranking[a,b] = top
#                 elif ((top == 3 and bot == 0)|(top == 0 and bot == 3)):  #prefered + not prefered = incomparable 
#                     final_ranking[a,b] = 1
#                 elif ((top == 3 and bot == 2)|(top == 2 and bot == 3)):  # indiff + prefered = prefered
#                     final_ranking[a,b] = 3
#                 elif ((top == 2 and bot == 0)|(top == 0 and bot == 2)):  #(added) indiff + not prefered = not prefered
#                     final_ranking[a,b] = 0
#                 else: 
#                     final_ranking[a,b] = 1
#                     
# =============================================================================
    
    end = timeit.default_timer()
    time_electre = end - start
    
    #electreiii.description = "Of a to b, 3 means prefered, 2 means indifferent, 1 means incompatible, and 0 means not prefered over. Remember that for prelimenary rankings that Python counts from 0"
    electreiii.time = time_electre
    electreiii.score = S_temp
    #electreiii.top = D_distil_rank
    #electreiii.bottom = A_distil_rank
    #electreiii.final = final_ranking
    return(electreiii)
    



