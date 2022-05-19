#ID function
number_of_requests = 1000
def ID_create(number_of_requests):
    import numpy as np
    digits = np.log10(number_of_requests) + 1
    list_id = list(range(1,number_of_requests+1))
    str_id = list(map(str, list_id))
    add_str = int(digits-1) * "0"
    str2_id = [[add_str + x][0][-4:] for x in str_id]
    return(str2_id)

