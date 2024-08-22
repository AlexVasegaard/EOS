import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

# Load the dataset (assuming it's a CSV file)
df = pd.read_csv("worldcities.csv")  #from https://simplemaps.com/data/world-cities


# Display the first few rows of the dataframe to understand its structure
print(df.head())

# Handle NaN population values by filling them with a small value (e.g., 1) to allow for sampling
df['population'] = df['population'].fillna(1)

# Normalize the population values to use as weights for sampling
df['sampling_weight'] = df['population'] / df['population'].sum()

# Sample 100,000 cities based on the population weight
sampled_df = df.sample(n=100000, weights='sampling_weight', replace=True, random_state=42)

# Function to add noise to the coordinates
def add_noise_to_coordinates(lat, long, population):
    # Adding noise inversely proportional to the square root of the population size
    noise_factor = 1 / np.sqrt(population + 1)  # Adding 1 to avoid division by zero
    lat_noise = np.random.uniform(-noise_factor, noise_factor)
    long_noise = np.random.uniform(-noise_factor, noise_factor)
    
    # Apply noise and clamp the values to ensure they are within valid ranges
    lat = np.clip(lat + lat_noise, -90, 90)
    long = np.clip(long + long_noise, -180, 180)
    
    return lat, long

# Create a new dataframe to store the satellite image acquisition requests
requests_df = pd.DataFrame()

# Apply the noise function to each city's latitude and longitude
requests_df['latitude'], requests_df['longitude'] = zip(*sampled_df.apply(
    lambda row: add_noise_to_coordinates(row['lat'], row['lng'], row['population']), axis=1))

# Directly assign city and country columns from the sampled data
requests_df['city'] = sampled_df['city'].values
requests_df['country'] = sampled_df['country'].values


# Save the requests to a new CSV file
requests_df.to_csv('worldloc.xlsx', index=False, header = True)
# Load the dataset (assuming it's a CSV file)
#requests_df = pd.read_csv("worldloc.xlsx")

print("Satellite image acquisition requests have been generated and saved to 'worldloc.csv'.")


# Plotting the sampled cities on a world map
# Load the world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Create a GeoDataFrame with the sampled cities
gdf = gpd.GeoDataFrame(requests_df, geometry=gpd.points_from_xy(requests_df.longitude, requests_df.latitude))

# Plot the world map
fig, ax = plt.subplots(figsize=(15, 10))
world.plot(ax=ax, color='lightgray')

# Plot the sampled cities
gdf.plot(ax=ax, color='red', markersize=1, alpha=0.5)

# Set plot title and labels
plt.title('Satellite Image Acquisition Requests locations', fontsize=15)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)

# Show the plot
plt.show()