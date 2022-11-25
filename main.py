import streamlit as st
import geocoder as gc
import requests
from base64 import encodebytes
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests


st.title("Billboard Location Predictor (Delhi)")
st.text("This model will predict the best possible locations for your category of bussiness.")
catergory = st.sidebar.selectbox(
    "Type Of Bussiness",
    ("Automobile", "Electronics", "F & B", "Media")
)
# 0-for Automobile 1- for F&B 2-Electronics 3- for Media
if catergory == "Automobile":
    choice_user = 0
elif catergory == "F & B":
    choice_user = 1
elif catergory == "Electronics":
    choice_user = 2
else:
    choice_user = 3

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests
import geocoder
from geopy.geocoders import Nominatim
import folium
from pandas.io.json import json_normalize
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import seaborn as sns
from sklearn.cluster import KMeans
from pandas.plotting import table

tehsil = pd.read_csv('delhi_Locality.csv')

tehsil.rename(columns = {"place_name": "Neighborhood", "admin_name3": "Borough", "latitude": "Latitude", "longitude": "Longitude"}, inplace = True)



latitude = 28.7041
longitude = 77.1025


CLIENT_ID = 'SMW0KBR1QLGVOEBX3HFOQFVCABUB5HXWFTUZZCZMHFD5PVTW' 
CLIENT_SECRET = '3KAMZFVCZ15XQ04SVTKVDMVTEFQJASZNQRSVTUUNUZHX4PDZ'
VERSION = '20200406'


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            1020, 
            100)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


def getTopLocations(f):  
  columns = ['Neighborhood', 'Common Venue', 'Advertisement']

  # create a new dataframe
  tehsil_venues_sorted = pd.DataFrame(columns=columns)
  tehsil_venues_sorted['Neighborhood'] = f['Neighborhood']
  tehsil_venues_sorted.dropna()
  for ind in range(f.shape[0]):
    tehsil_venues_sorted.iloc[ind, 1:] = return_most_common_venues(f.iloc[ind, :], 1)
  return tehsil_venues_sorted


def makeClusters(criteria):
  k_rng = range(1,10)
  sse = []
  # f = frequency[['Neighborhood', 'Bank', 'High School', 'Shopping Mall', 'Train Station']]
  for k in k_rng:
      km = KMeans(n_clusters=k)
      km.fit(criteria)
      # km.fit(frequency[['Bank', 'High School', 'Shopping Mall', 'Train Station']])
      #km.fit(f)
      sse.append(km.inertia_)
  
#   plt.xlabel('k')
#   plt.ylabel('Sum of Squared error')
#   plt.plot(k_rng,sse)

from re import X
def clustering(tvs, size, ad_class):
  kclusters = size
  tehsil_grouped_clustering = tvs.drop('Neighborhood', 1)

  x = tehsil_grouped_clustering[['Common Venue']]
  target = tehsil_grouped_clustering[['Advertisement']]

  kmeans = KMeans(n_clusters=kclusters).fit(x)
  kmeans.labels_[0:10]
  

  tvs.insert(0, 'Cluster Labels', kmeans.labels_)
  tehsil_merged = tehsil
  tehsil_merged = tehsil_merged.join(tehsil_venues_sorted.set_index('Neighborhood'), on='Neighborhood')
  temp = tehsil_merged
  temp = temp[temp['Cluster Labels'].notna()]

  map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)
  # set color scheme for the clusters
  x = np.arange(kclusters)
  ys = [i + x + (i*x)**2 for i in range(kclusters)]
  colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
  rainbow = [colors.rgb2hex(i) for i in colors_array]

  # add markers to the map
  markers_colors = []
  areas = []
  for lat, lon, poi, cluster in zip(temp['Latitude'], temp['Longitude'], temp['Neighborhood'], temp['Cluster Labels']):
      #if cluster != undefined:
      cluster = int(cluster)
      cluster1 = tehsil_merged.loc[tehsil_merged['Cluster Labels'] == cluster, tehsil_merged.columns[[1] + list(range(5, tehsil_merged.shape[1]))]]
      if ad_class not in cluster1['Advertisement'].values:
        continue
      areas.append(cluster1['Neighborhood'].tolist())
      label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
      folium.CircleMarker(
          [lat, lon],
          radius=25,
          popup=label,
          color=rainbow[cluster-1],
          fill=True,
          fill_color=rainbow[cluster-1],
          fill_opacity=0.7).add_to(map_clusters)

  #areas = set(areas)
  #cls = set(cls)
  a_list = []
  for area in areas:
    for a in area:
      a_list.append(a)


  return map_clusters


if __name__ == '__main__':


    neighborhood_latitude = tehsil.loc[1, 'Latitude'] # Tehsil latitude value
    neighborhood_longitude = tehsil.loc[1, 'Longitude'] # Tehsil longitude value

    neighborhood_name = tehsil.loc[1, 'Neighborhood']# Tehsil name


    url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
        CLIENT_ID, 
        CLIENT_SECRET, 
        VERSION, 
        neighborhood_latitude, 
        neighborhood_longitude, 
        1020, 
        100)



    n_venues = getNearbyVenues(names=tehsil['Neighborhood'],latitudes=tehsil['Latitude'], longitudes=tehsil['Longitude'])
    frequency = pd.get_dummies(n_venues[['Venue Category']], prefix="", prefix_sep="")

    # add neighborhood column back to dataframe
    frequency['Neighborhood'] = n_venues['Neighborhood'] 
    fixed_columns = [frequency.columns[-1]] + list(frequency.columns[:-1])
    frequency = frequency[fixed_columns]

    frequency_grouped = frequency.groupby('Neighborhood').mean().reset_index()
    f = frequency_grouped
    

    tehsil_venues_sorted = getTopLocations(f)

    
    tehsil_venues_sorted.loc[tehsil_venues_sorted['Common Venue'] == 'Train Station', 'Advertisement'] = 'Automobile'
    tehsil_venues_sorted.loc[tehsil_venues_sorted['Common Venue'] == 'Airport Terminal', 'Advertisement'] = 'Automobile'
    tehsil_venues_sorted.loc[tehsil_venues_sorted['Common Venue'] == 'Hotel', 'Advertisement'] = 'Automobile'

    tehsil_venues_sorted.loc[tehsil_venues_sorted['Common Venue'] == 'Indian Restaurant', 'Advertisement'] = 'F & B'
    tehsil_venues_sorted.loc[tehsil_venues_sorted['Common Venue'] == 'Udupi Restaurant', 'Advertisement'] = 'F & B'
    tehsil_venues_sorted.loc[tehsil_venues_sorted['Common Venue'] == 'Farm', 'Advertisement'] = 'F & B'
    tehsil_venues_sorted.loc[tehsil_venues_sorted['Common Venue'] == 'Fast Food Restaurant', 'Advertisement'] = 'F & B'
    tehsil_venues_sorted.loc[tehsil_venues_sorted['Common Venue'] == 'Pizza Place', 'Advertisement'] = 'F & B'

    tehsil_venues_sorted.loc[tehsil_venues_sorted['Common Venue'] == 'ATM', 'Advertisement'] = 'Electronics'
    tehsil_venues_sorted.loc[tehsil_venues_sorted['Common Venue'] == 'Shop & Service', 'Advertisement'] = 'Electronics'

    tehsil_venues_sorted.loc[tehsil_venues_sorted['Common Venue'] == 'Cosmetics Shop', 'Advertisement'] = 'Media'
    tehsil_venues_sorted.loc[tehsil_venues_sorted['Common Venue'] == 'Playground', 'Advertisement'] = 'Media'

    makeClusters(f.loc[:, f.columns != 'Neighborhood'])

    from sklearn import preprocessing
    import io
    from PIL import Image

    le = preprocessing.LabelEncoder()
    temp = tehsil_venues_sorted
    target = temp['Common Venue']
    le.fit(target.tolist())
    tvs = le.transform(target)
    temp['Common Venue'] = pd.Series(tvs)

    target = temp['Advertisement']
    le.fit(target.tolist())
    tvs = le.transform(target)
    temp['Advertisement'] = pd.Series(tvs)
    # 0-for Automobile 1- for F&B 2-Electronics 3- for Media
    map_delhi = clustering(temp, 9, choice_user)
    from streamlit_folium import folium_static 
    folium_static(map_delhi)