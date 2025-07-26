'''
in linear_baseline, you MUST pass the shape file in counties or it will not work
do not pass TreeNode structures either, just the county
'''

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString

def linear_baseline(starting_county, end_county, counties, num_unk=1, plot=False):
    try:
        start_point = getCOUNTYcentroid(starting_county, counties).geometry.iloc[0]
        end_point = getCOUNTYcentroid(end_county, counties).geometry.iloc[0]
      
        line = LineString([start_point, end_point])
        intermediate_points = []
        for i in range(1, num_unk + 1):
            fraction = i / (num_unk + 1)
            point = line.interpolate(fraction, normalized=True)  
            intermediate_points.append(point)
        
        intermediate_gdf = gpd.GeoDataFrame(geometry=intermediate_points, crs=counties.crs)
        joined = gpd.sjoin(intermediate_gdf, counties, how="left", predicate="within")
        intermediate_counties = joined["GEOID"].tolist() 
        return intermediate_counties
        
    except (IndexError, KeyError, AttributeError) as e:
        print(f"Warning: Skipping interpolation between {starting_county} and {end_county}. Error: {type(e).__name__}")
        return []

    if plot:
        line_gdf = gpd.GeoDataFrame(geometry=[line], crs=counties.crs)
        ax = counties.plot(edgecolor='gray', facecolor='none', figsize=(10, 10))
        gpd.GeoDataFrame(geometry=[start_point], crs=counties.crs).plot(ax=ax, color='red', markersize=60)
        gpd.GeoDataFrame(geometry=[end_point], crs=counties.crs).plot(ax=ax, color='red', markersize=60)
        line_gdf.plot(ax=ax, color='black', linewidth=2)
        intermediate_gdf.plot(ax=ax, color='green', markersize=40)
        
    
        plt.title(f"Interpolation of {num_unk} points between {starting_county} and {end_county}")
        plt.show()
        
    joined = gpd.sjoin(intermediate_gdf, counties, how="left", predicate="within")
    intermediate_counties = joined["GEOID"].tolist() 
    return intermediate_counties
    

def getCOUNTYcentroid(geoid, counties):
    county = counties[counties["GEOID"] == geoid]
    if county.empty:
        raise KeyError(f"County with GEOID '{geoid}' not found")
    county_centroid = county.geometry.centroid.iloc[0]  
    return gpd.GeoDataFrame(geometry=[county_centroid], crs=counties.crs)


def plot(geoids, counties):
    ax = counties.plot(edgecolor='gray', facecolor='none', figsize=(10, 10))
    for p in geoids:
        point_gdf = getCOUNTYcentroid(int(p), counties)
        point_gdf.plot(ax=ax, color='red', markersize=100)
    plt.show()
