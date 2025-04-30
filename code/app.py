import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.title("E-commerce Customer Segmentation")

st.write("Upload your e-commerce dataset to perform customer segmentation.")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  
    st.write(df.head())  
    
    # Preprocessing
    df = df.drop(columns=["CustomerID"], errors="ignore")
    df = df.dropna()
    scaler = StandardScaler()
    numeric_df = df.select_dtypes(include=['float64', 'int64']) 
    df_scaled = scaler.fit_transform(numeric_df)

    
    # Clustering
    kmeans = KMeans(n_clusters=4, init="k-means++", random_state=42)
    df["Cluster"] = kmeans.fit_predict(df_scaled)
    
    st.write("Clustered Data:")
    st.write(df.head())
    
    # Visualization
    fig, ax = plt.subplots()
    sns.scatterplot(x=df["Annual Income (k$)"], y=df["Spending Score (1-100)"], hue=df["Cluster"], palette="viridis", ax=ax)
    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Spending Score")
    st.pyplot(fig)
