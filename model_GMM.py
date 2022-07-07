#app for the 2 options : input parameters or load data file

import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns 

scaler = StandardScaler()

st.image("http://www.ehtp.ac.ma/images/lo.png")
st.write("""
## MSDE4 : RFM Clustering Prediction Application
###This app Define the **Clients** Cluster 
""")

st.sidebar.image("https://sarahtianhua.files.wordpress.com/2015/09/rfm-300x278.png",width=300)

option = st.selectbox(
     'How would you like to use the prediction model?',
     ('','input parameters directly', 'Load a file of data'))

def user_input_features():
    Recency = st.sidebar.slider('Recency', min_value=0,max_value=374)
    Frequency = st.sidebar.slider('Frequency', min_value=0,max_value=7847)
    MonetaryValue = st.sidebar.slider('MonetaryValue in €', min_value=0,max_value=280206)
    data = {
    	    'Recency': -1+(Recency*(2)/374),
            'Frequency': -1+(Frequency*(2)/7847),
            'Monetary Value': -1+(MonetaryValue*(2)/280206),
            }

    features = pd.DataFrame(data, index=[0])
    return features

def show_results1():
    st.subheader('User Input Client Features: ')
    st.write(df) 
    model_gmm = pickle.load(open("model_GMM_RFM.pkl", "rb"))
    prediction = model_gmm.predict(df)    

    st.subheader('Cluster du Client : ')    
    st.write(prediction)

    if prediction == 0:
            st.write("Cluster 0 : Meilleurs Clients de la boite, avec une grande fréquence d’achats  et un chiffre d’affaires conséquent et des commandes récentes.")
            st.write("Un Client fidèle et representant Presque les 80% du chiffre d’affaire global, un client qui doit être orienté vers un traitement long term reposant sur des contrats de long durée et des offres de prix préférentielles.")
    elif prediction == 1:
        st.write("Cluster 1 : Clients avec des commandes anciènnes et qui ne commande plus assez, avec une fréquence et un chiffre d’affaire moyen.")
        st.write("Clients qui ont peut être cherché ailleurs pour leur commandes récente, l’objectif de l’equipe commerciale serais de les recontacter et savoir les causes de leurs changement, et par la suite essayer de les reconquerir avec des offres concurrentielles.")
    elif prediction == 2:
        st.write("Cluster 2 : Nouveaux Clients, avec des commandes récentes et des valeurs moyennes en fréquence en chiffre d’affaire.")
        st.write("Clients qui font leurs premières commandes et qui font connaissance des produits et du service de l’entreprise.Cette categorie doit être informé sur toute la gamme et doit être conseillé et orienté, pour maximiser son chiffre d’affaire et sa fréquence.")
 

def show_results2():

    st.subheader('User Input Client Features: ')
    st.write(df_gmm) 
    model_gmm = pickle.load(open("model_GMM_RFM.pkl", "rb"))
    prediction = model_gmm.predict(df)
    prediction = pd.DataFrame(prediction)
    st.subheader('Clusters des Clients : ') 
    df_gmm['clusters']=prediction.values
    st.write(df_gmm)

def show_plot():
    fig = plt.figure(figsize=(12,7), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.axes(projection="3d")
    ax.scatter3D(df.T[0],df.T[1],df.T[2],c = df_gmm.clusters.values ,cmap='Spectral')
    xLabel = ax.set_xlabel('Recency')
    yLabel = ax.set_ylabel('Frequency')
    zLabel = ax.set_zlabel('Monetary Value')

  
if option=='input parameters directly':
    st.sidebar.header('User Input Parameters')
    df = user_input_features()
    show_results1()    
    
elif option=='Load a file of data':
    uploaded_file = st.file_uploader("Choose an RFM file to load with Columns as follows :(CostumerID, Recency, Frequency, MonetaryValue) ")
    if uploaded_file is not None:
        df1 = pd.read_csv(uploaded_file)
        df = df1.iloc[1:,[1,2,3]].values
        df_gmm = pd.DataFrame({   
        'Recency': df.T[0],
        'Frequency': df.T[1],
        'MonetaryValue': df.T[2],
        })
        df_gmm['CustomerID'] = df1['CustomerID']
        df = scaler.fit_transform(df)
        df = normalize(df)
        show_results2()
        show_plot()
    
