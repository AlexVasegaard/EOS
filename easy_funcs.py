#this script contains the code computing both the random data and the  
#algorithms for AHP, TOPSIS, and ELECTRE. Note, it computes running time.

#rows = 2000
#cols = 20

#import numpy as np
#data = np.random.rand(rows, cols) #add extra dimension for fuzzy layer

#data = np.matrix([[25,10, 30],[20,30,10],[15,20,30],[30,30,10]])
#this means data is represented as - each coloum to be a possible area bid
#and each row is a criteria.. each element is then a score



## AHP ##
#overview - alternatives against each other for each criteria and criterias against each other
#we will get #rows 0 + 1 matrices. But computationally that would be to big a 
#deal, so we create a matrix for each criteria, normalize it, computes averages, 
#and inserts these in the overall decision matrix 

def AHP(data, weight_criteria = None):
    import timeit
    import numpy as np
    start = timeit.default_timer()
    data_overall = np.zeros((data.shape[1],data.shape[0]))
    data_temp = np.zeros((data.shape[1],data.shape[1]))
    #score for each alternative
    for i in range(0,data.shape[0]):
        #find score
        for j in range(0, data.shape[1]):
            for k in range(0, data.shape[1]):
                data_temp[j,k] = data[i,j] / data[i,k]
        #normalize
        for l in range(0, data.shape[1]):
            data_temp[:, l] = data_temp[:, l]/sum(data_temp[:, l])
        #average 
        for m in range(0, data.shape[1]):
            data_overall[m,i] = data_temp[m,:].mean() 
    #overall score computed with the weight for each criteria set by the dm 
    if weight_criteria == None:
        weight_criteria = [1]*data.shape[0]
        weight_criteria = np.asarray(weight_criteria)/sum(weight_criteria)

    ODscore = np.matmul(data_overall, weight_criteria)
    ODrank = np.argsort(ODscore)
    end = timeit.default_timer()
    time_AHP = end - start
    #note the last is the biggest (and begins from 0)
    AHP.rank = ODrank
    AHP.time = time_AHP
    return(AHP)




## TOPSIS ##

#topsis contains also multiple objectives, i.e. both maximization and 
#minimization of some criteria. A vector representing each criterias objective
#is therefore created.

#objective_data = np.random.randint(low = 0, high = 2, size = data.shape[0])
#objective_data = np.array([1,1,1,1,1])
#1 indicates a max objective, and 0 a minimisation objective

#topsis normalises data (also negative elements), weights the normalised data, 
#and creates a seperation measure from optimal worst and best solutions, and 
#then ranks the measures with regard to their relative closeness.

def topsis(data, objective_data, weight_criteria):
    import timeit
    import numpy as np
    start = timeit.default_timer()

    #assign weights and normalize them - should be done by DM. Here equal
    if any(weight_criteria == None):
        weight_criteria = [1]*data.shape[0]
        weight_criteria = np.asarray(weight_criteria)/sum(weight_criteria)

    #create empty dataframe
    data_WN = np.zeros((data.shape[0],data.shape[1]))
    #normalize and weigh the data 
    for i in range(0, data.shape[0]):
        data_WN[i,:] = (data[i,:]/np.sum(np.square(data[i,:]))**0.5) * weight_criteria[i]
    #find ideal best and worst solutions (max and min depending on objective for 
    #each criteira). Assign these in new df. note, 2 is one for both best and worst 
    ideal_criteria = np.zeros((data.shape[0], 2))
    for j in range(0, data.shape[0]):
        if objective_data[j] == 1:
            ideal_criteria[j,0] = max(data_WN[j,:])
            ideal_criteria[j,1] = min(data_WN[j,:])
        else:
            ideal_criteria[j,0] = min(data_WN[j,:])
            ideal_criteria[j,1] = max(data_WN[j,:])
    #compute seperation measure
    seperation_matrix = np.zeros((2,data.shape[1]))
    for k in range(0, data.shape[1]):
        seperation_matrix[:,k] = np.sqrt(np.sum(np.square(np.array([data_WN[:,k],data_WN[:,k]]).transpose()-ideal_criteria), axis = 0))
    
    #measure relative closeness i.e. worst/(worst+best)
    relative_closeness = np.zeros(data.shape[1])
    for l in range(0, data.shape[1]):
        relative_closeness[l] = seperation_matrix[1,l]/np.sum(seperation_matrix[:,l])
    #score is basically the length to the worst, so the higher value the better!
    rank = np.argsort(relative_closeness)
    end = timeit.default_timer()
    time_topsis = end - start
    
    topsis.score = relative_closeness
    topsis.rank = rank
    topsis.time = time_topsis
    return(topsis)
    
    


## ELECTRE ##
#will propably let way to many alternatives be better than the others..

#electre normalizes and weighs the data. It then deals with the issue of 
#outranking in the regard of concordance and discordance set, i.e. it tries
#to find incentive that ranks alternatives with one over another.

def electre(data,  weight_criteria = None):
    import timeit
    import numpy as np
    start = timeit.default_timer()
    #assign weights and normalize them - should be done by DM. Here equal
    if weight_criteria.all() == None:
        weight_criteria = [1]*data.shape[0]
        weight_criteria = np.asarray(weight_criteria)/sum(weight_criteria)
        #weight_criteria =np.array([0.2,0.15,0.4,0.25])

    #create empty dataframe
    data_WN = np.zeros((data.shape[0],data.shape[1]))
    #normalize and weigh the data 
    for n in range(0, data.shape[0]):
        data_WN[n,:] = (data[n,:]/np.sum(np.square(data[n,:]))**0.5) * weight_criteria[n]

    #concordance and discordance set, setup concordance and discordance matrix
    CM_temp = np.zeros((data.shape[1],data.shape[1]))
    DM_temp = np.zeros((data.shape[1],data.shape[1]))
    for i in range(0, data.shape[1]):
        for j in range(0, data.shape[1]):
            if i==j:
                CM_temp[i,j] = 0
                DM_temp[i,j] = 0
            else:
                diff_vector = data_WN[:,i] - data_WN[:,j]
                try:
                    CM_temp[i,j] = np.sum(weight_criteria[diff_vector>=0])
                except:
                    CM_temp[i,j] = 0
                DM_temp[i,j] = abs(min(diff_vector))/max(abs(diff_vector))
    #calculate C and D bar
    C_bar = np.sum(CM_temp)/(data.shape[1]**2-data.shape[1])
    D_bar = np.sum(DM_temp)/(data.shape[1]**2-data.shape[1])
    #check wheter CM > c bar and DM > d bar
    CM_final = (CM_temp>C_bar)*1
    DM_final = (DM_temp>D_bar)*1
    
    ODM = np.multiply(CM_final, DM_final)
    
    end = timeit.default_timer()
    time_electre = end - start
    
    electre.rank = ODM
    electre.time = time_electre
    return(electre)

#meaning in text
#f = np.sum(ODM)
#for e in range(0,f):
#    txt1 = "this means alternative"
#    i = np.where(ODM==1)[e]
#    txt2 = "is greater than"
#    print(txt1, i[0]+1, txt2, i[1]+1)


