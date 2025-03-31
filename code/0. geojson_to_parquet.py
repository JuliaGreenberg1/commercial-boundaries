import geopandas as gpd

# Load the GeoJSON file
gdf = gpd.read_file("/Users/jpg23/git/urban-activity-atlas/data/metro_regions_full.geojson")

# Save it as Parquet
gdf.to_parquet("/Users/jpg23/data/downtownrecovery/commercial_districts/commercial_districts_paper/top300metros.parquet")