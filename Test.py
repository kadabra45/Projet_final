import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import time
from pprint import pprint
import streamlit as st
import csv
import io
import PIL 
from PIL import Image
from surprise import SVD
from surprise.dataset import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler 
import folium
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LogisticRegression
import requests, json
import urllib.parse
import  pyautogui
import cv2
from selenium import webdriver
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary

sns.set(style="white") #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)

# binary = FirefoxBinary("C:/Program Files/Mozilla Firefox/firefox.exe")
# browser = webdriver.Firefox()
# Utilisation du navigateur Firefox pour la recherche d'informations sur Internet.

localisation = []
genre = []
stade_dev = []
arr = []
adresse = []

with open('les-arbres.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile, delimiter=';')
    for row in csvReader:
        localisation.append(row[16])
        genre.append(row[9])
        stade_dev.append(row[14])
        arr.append(row[3])
        adresse.append(row[6])

df_stade_dev = pd.DataFrame(stade_dev)
df_genre = pd.DataFrame(genre)
df_localisation = pd.DataFrame(localisation)
df_arr = pd.DataFrame(arr)
df_adresse = pd.DataFrame(adresse)

st.title("Géolocalisation des zones allergisantes")
# Création du titre de l'application Streamlit.

df_adresse.rename(columns={0:"LIEU/ADRESSE"}, inplace=True)
df_arr.rename(columns={0:"ARRONDISSEMENT"}, inplace=True)
df_genre.rename(columns={0:"GENRE"}, inplace=True)
df_localisation.rename(columns={0:"geo_point_2d"}, inplace=True)
df_stade_dev.rename(columns={0:"STADE"}, inplace=True)

df = df_arr.join(df_genre)
df2 = df.join(df_adresse)
df3 = df2.join(df_localisation)
df4 = df3.join(df_stade_dev)

df4 = df4.drop(0)

df4['GENRE'] = df4['GENRE'].map(lambda x: x.lower())
df4['ARRONDISSEMENT'] = df4['ARRONDISSEMENT'].map(lambda x: x.lower())
df4['LIEU/ADRESSE'] = df4['LIEU/ADRESSE'].map(lambda x: x.lower())
df4['STADE'] = df4['STADE'].map(lambda x: x.lower())

df4.loc[df4["GENRE"]=="betula","Allergisant"] = "1"
df4.loc[df4["GENRE"]=="cupressus","Allergisant"] = "1"
df4.loc[df4["GENRE"]=="corylus","Allergisant"] = "1"
df4.loc[df4["GENRE"]=="platanus","Allergisant"] = "1"
df4.loc[df4["GENRE"]=="olea","Allergisant"] = "1"
df4.loc[df4["GENRE"]=="fraxinus","Allergisant"] = "1"
df4.loc[df4["GENRE"]=="quercus","Allergisant"] = "1"
df4.loc[df4["GENRE"]=="carpinus betulus","Allergisant"] = "1"
df4.loc[df4["Allergisant"]!="1","Allergisant"] = "0"

df4.loc[df4['GENRE'].str.contains("betula"),'GENRE']='1'
df4.loc[df4['GENRE'].str.contains("cupressus"),'GENRE']='2'
df4.loc[df4['GENRE'].str.contains("corylus"),'GENRE']='3'
df4.loc[df4['GENRE'].str.contains("platanus"),'GENRE']='4'
df4.loc[df4['GENRE'].str.contains("olea"),'GENRE']='5'
df4.loc[df4['GENRE'].str.contains("fraxinus"),'GENRE']='6'
df4.loc[df4['GENRE'].str.contains("quercus"),'GENRE']='7'
df4.loc[df4['GENRE'].str.contains("carpinus betulus"),'GENRE']='8'

df4.loc[(df4["Allergisant"]=="1") & (df4["STADE"]!="ja"), "ALLERGIE"] = "oui"
df4.loc[(df4["Allergisant"]=="0") | (df4["STADE"]=="ja"), "ALLERGIE"] = "non"

encoder=LabelEncoder()
df4['ALLERGIE']=encoder.fit_transform(df4['ALLERGIE'])

df5 = pd.DataFrame(df4.geo_point_2d.str.split(",",1).tolist(), columns = ['LATITUDE','LONGITUDE'])

df5= df5.astype(float)

del df4['geo_point_2d']

df_f = df4.join(df5)

df_i = df_f.loc[df_f["ARRONDISSEMENT"].str.contains('paris') == True]

df_i = df_i.drop(204726)

df_i['ARRONDISSEMENT']=encoder.fit_transform(df_i['ARRONDISSEMENT'])  
df_i['LIEU/ADRESSE']=encoder.fit_transform(df_i['LIEU/ADRESSE']) 
df_i['STADE']=encoder.fit_transform(df_i['STADE'])  
df_a = df_i.loc[df_i["ALLERGIE"] == 1]

reader = Reader(rating_scale=(1, 164151))
df_etude_2 = Dataset.load_from_df(df_a[['LATITUDE', 'LONGITUDE','GENRE']], reader)

X = StandardScaler().fit_transform(df_a)
algo5 = DBSCAN(eps=0.3, min_samples=7).fit(X)
labels = algo5.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

train_2, test_2 = train_test_split(df_etude_2, test_size=.25)
algo = SVD()
predictions_2 = algo.fit(train_2).test(test_2)

lat = []
lng = []

for i in predictions_2:
    lat.append(i[0])
    lng.append(i[1])

df_lat = pd.DataFrame(lat)
df_lng = pd.DataFrame(lng)

df_lat.rename(columns={0:"lat"}, inplace=True)
df_lng.rename(columns={0:"lon"}, inplace=True)

df = df_lat.join(df_lng)

coords = (48.864716,2.349014)
map = folium.Map(location=coords, tiles='OpenStreetMap')

kmeans = KMeans(n_clusters=n_clusters_, random_state=0).fit(df)

for la, ln in zip(lat, lng):
    folium.CircleMarker(
        location=[la, ln],
    ).add_to(map)

for l in kmeans.cluster_centers_:
    folium.CircleMarker(
        location=l,
        color='red'
    ).add_to(map)

# Le code ci-après concerne la mise en place de l'application web par Streamlit.
input_box_1 = st.text_input("Lieu dans Paris : ", value='', max_chars=None, key=None, type='default')
# input_box_1 correspond à la barre pour rentrer l'adresse dans l'application.

api_url = "https://api-adresse.data.gouv.fr/search/?q="
r = requests.get(api_url + urllib.parse.quote(input_box_1))
# r correspond aux informations géographiques obtenues par l'url.

app = Nominatim(user_agent="tutorial")

def get_location_by_address(address):
    """This function returns a location as raw from an address
    will repeat until success"""
    time.sleep(1)
    return app.geocode(address)
    # except:
    #     return get_location_by_address(address)


location_res = get_location_by_address(input_box_1)

if location_res:
    lat = location_res.latitude
    lon = location_res.longitude
    folium.Marker(
            location=[lat, lon],
            color='blue'
        ).add_to(map)
# Création du marker de localisation de l'adresse entrée dans l'application.

    # output = open("file01.jpg","wb")
    # output.write(map)
    # st.image(output)
    st.map(df, zoom = 10)
# Création de la map par Streamlit