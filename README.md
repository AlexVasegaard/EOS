 # EOS
EOS is a set of functions (intended to be a package) that encompases everything within a centralized Earth Observation Satellite scheduling system in terms of Scenario generation, pre-processing, problem setup, solution approach, decision maker preference integration, and visualization.

The DM preferences are introduced through a plethora of Scoring approaches available:
- modified ELECTRE-III (ordinal)
- topsis (ordinal)
- WSA (ordinal)
all scoring approaches contain representative variables to elicited information as opposed to using pairwise evaluations.

![alt text](single_scenario_map.PNG)

the two main functions are multi_sat_data() and multi_sat_testing(). 

the package dependencies are:
- numpy, pandas, datetime, requests, random, ephem, math, folium (for a visual html map output), time, scipy, progressbar, ast, timeit 

and depending on whether a free optimization method is used (api may be required):
- cvxopt, gurobipy, pulp, docplex

Real satellite paths are introduced trough their TLE (Go to www.celestrak.com to obtain TLEs, default are Spot 6,7 and Pleiades A and B)
Also, there is an option to obtain realtime, historic, or generate weather data (cloud coverage) when generating the scenario. 

## multi_sat_data() 
Generates the problem, so it functions as a general pre-processing for the EOS system. 
It is seeded so problem scenarios can be replicated across different environments and therefore utilized for evaluating different solution approaches.
Note, it isnt optimized for speed yet, so it will run rather slow.

It takes in the following arguments: 
- seconds_gran = 20 %The discretisation level of the satellitel path (discrete optimization problem) 
- number_of_requests_0 = 1000, %customer requests in database initially (there is an option to contionously add customers to mimic the effect of a real EOS production where new customers are entering and one over time still wants to ensure that requests doesnt violate an age threshold) 
- NORAD_ids = [38755, 40053]  %TLEs for spot 6 and 7 satellites
- weather_real = False, %whether real cloud coverage data is utilized for the chosen time horizon
- simplify = False, #whether constraints are simplified based on the principle of inter set constraints
- schedule_start = [2021,7,21,9,40],  %time of initiation for the schedule horizon
- hours_horizon = 8, %duration of planning horizon in hours
- max_off_nadir_angle = 30, %degrees that satellite can maneuver (or is allowed to still acquire pictures) 
- height_satellite = 694,   %altitude of satellites (in km) - this is in next iteration updated to automatically be calculated 
- rotation_speed = 30/12, %degrees per second - per https://directory.eoportal.org/web/eoportal/satellite-missions/s/spot-6-7
- cam_resolution = 1, %m^2 per pixel
- capacity_limit = 1000000, %in mega byte
- satellite_swath = 3600, &swath of satellite images 
- map_generation = True %whether a visualisation should be generated

AND outputs the following:
 - multi_sat_data.LPP is the Linear programming problem Ax<=b where LPP contains:
   - LPP.LHS - A in the Ax<b
   - LPP.RHS - b in the Ax<b
   - LPP.eLHS - A in the Ax=b
   - LPP.eRHS - b in the Ax=b
 - multi_sat_data.df is the data frame containing all information for the entire problem scenario (for each attempt)
 - multi_sat_data.pf_df is the performance data frame for the relevant (reachable) image attempts
 - multi_sat_data.m is the folium map with relevant problem scenario information

## multi_sat_testing() 
This function contains both the preference integration part (scoring) and the solution approach.
It takes in the following arguments:
- scoring_method (can be 1 = TOPSIS, 2 = ELECTRE, 3 = naive scoring method WSA)
- solution_method (can be "gurobi", "PuLP", "cplex", or "DAG")  
- criteria_weights (relevant for TOPSIS, ELECTRE, and WSA), e.g. np.array([1,0,1,0,0,0,1,1,1])
- threshold_parameters (relevant for ELECTRE), e.g. np.array([[0, 2, 2, 0, 0,0,0,0],[50,5,5,5,1, 1000, 0, 2], [1000, 40, 40, 15, 2, 10000, 13, 5]])
- alpha a scalar, it is the factor with which scores are taken to the power of. It basically represent the level with which one trusts the average score - it supplies the DM with ratio evaluation ability. Default value is 1 meaning this is negleted.

Note, the order with which criteria are presented in the criteria weights and threshold_parameters arguments are:
- area, 
- angle, 
- sun elevation, 
- cloud cover, 
- priority, 
- price, 
- age, 
- uncertainty

AND outputs the following:
 - multi_sat_testing.x is the binary solution vector illustrating which attempts should be acquired and which should be neglected
 - multi_sat_testing.score is the generated score for each attempt through the introduced preference setting
 - multi_sat_testing.time is the runtime for the solution approach

### PLEASE let me know if you have any suggestions (good or bad) to the code - any comments are highly appreciated :-) 
