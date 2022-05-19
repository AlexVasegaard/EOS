#plot all requests
#adding the requests with color dependent on number of strips
def color_strips(strips):
    if strips == 1:
        color = "gray"
    if strips == 2:
        color = "royalblue"
    if strips == 3:
        color = "blue"
    if strips > 3:
        color = "navy"
    return(color)
    
def plot_requests(m, df, name, radius):
    import folium
    #import pandas as pd
    import numpy as np
    
    ar = folium.FeatureGroup(name = "all requests")
    
    #plot requests as marker in folium map
    #for each row in the data, add a cicle marker
    for i in range(0,df.shape[0]):
        #generate the popup message that is shown on click.
        popup_text = "{}<br> ID: {}<br> area: {}<br> stereo: {}<br> strips*: {}<br> duration: {}<br> priority: {}<br> price: {}<br> waiting time: {}"
        popup_text = popup_text.format(df.iloc[i][name],
                                       df.iloc[i]["ID"],
                                       df.iloc[i]["area"],
                                       df.iloc[i]["stereo"],
                                       df.iloc[i]["strips"],
                                       int(df.iloc[i]["duration"]),
                                       df.iloc[i]["priority"],
                                       df.iloc[i]["price"],
                                       df.iloc[i]["waiting time"])
                
        # add marker to the map
        area_s = np.sqrt(df.iloc[i]["area"])/2 #one side in the assumed square area
        #convert km to degrees
        latlon_add = area_s/111.03  # assumed spherical earth
        loc = df.iloc[i][name]
        if df.iloc[i]["stereo"] == 1:
            color1 = color_strips(df.iloc[i]["strips"])
        else:
            color1 = "black" #violet
        ar.add_child(folium.PolyLine([[loc[0]+latlon_add,loc[1]-latlon_add],[loc[0]+latlon_add,loc[1]+latlon_add],
                                      [loc[0]-latlon_add,loc[1]+latlon_add],[loc[0]-latlon_add,loc[1]-latlon_add],
                                      [loc[0]+latlon_add,loc[1]-latlon_add]], popup=folium.Popup(popup_text), color=color1, weight=8))
    
        #folium.CircleMarker(location=df.iloc[i][name],
        #                    radius = radius, color = color, popup = popup_text).add_to(m)
    m.add_child(ar)
    #m.add_child(folium.LayerControl())
    m.save("all_requests.html")

def color_priority(priority):
    if priority == 1:
        color = "darkviolet"
    if priority == 2:
        color = "crimson"
    if priority == 3:
        color = "red"
    if priority == 4:
        color = "orangered"
    if priority == 5:
        color = "orange"
    if priority == 6:
        color = "yellow"
    if priority == 7:
        color = "white"
    return(color)


#adding the request with color depending on which type of priority (performance_df)
#performance_df.info()    
def plot_requests2(m, df, name):
    import folium
    #import pandas as pd
    import numpy as np
    
    rr = folium.FeatureGroup(name = "feasible requests")
    
    #plot requests as marker in folium map
    #for each row in the data, add a cicle marker
    for i in range(0,df.shape[0]):
        #generate the popup message that is shown on click.
        popup_text = "{}<br> ID: {}<br> area: {}<br> stereo: {}<br> strips*: {}<br> duration: {}<br> priority: {}<br> price: {}<br> waiting time: {}"
        popup_text = popup_text.format(df.iloc[i][name],
                                       df.iloc[i]["ID"],
                                       df.iloc[i]["area"],
                                       df.iloc[i]["stereo"],
                                       df.iloc[i]["strips"],
                                       int(df.iloc[i]["duration"]),
                                       df.iloc[i]["priority"],
                                       df.iloc[i]["price"],
                                       df.iloc[i]["waiting time"])
                
        # add marker to the map
        area_s = np.sqrt(df.iloc[i]["area"])/2 #one side in the assumed square area
        #convert km to degrees
        latlon_add = area_s/111.03  # assumed spherical earth
        loc = df.iloc[i][name]
        color2 = color_priority(df.iloc[i]["priority"])
        rr.add_child(folium.PolyLine([[loc[0]+latlon_add,loc[1]-latlon_add],[loc[0]+latlon_add,loc[1]+latlon_add],
                                      [loc[0]-latlon_add,loc[1]+latlon_add],[loc[0]-latlon_add,loc[1]-latlon_add],
                                      [loc[0]+latlon_add,loc[1]-latlon_add]], popup=folium.Popup(popup_text), color=color2, weight=8))
        
        #folium.CircleMarker(location=df.iloc[i][name],
        #                    radius = radius, color = color, popup = popup_text).add_to(m)
    m.add_child(rr)
    m.add_child(folium.LayerControl())
    m.save("all_requests.html")