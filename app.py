import streamlit as st

st.title("E-commerce Customer Segmentation")

st.write("Upload your e-commerce dataset to perform customer segmentation.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Preprocessing
    df = df.drop(columns=["CustomerID"], errors="ignore")
    df = df.dropna()
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    # Clustering
    kmeans = KMeans(n_clusters=4, init="k-means++", random_state=42)
    df["Cluster"] = kmeans.fit_predict(df_scaled)
    
    st.write("Clustered Data:")
    st.write(df.head())
    
    # Visualization
    fig, ax = plt.subplots()
    sns.scatterplot(x=df["AnnualIncome"], y=df["SpendingScore"], hue=df["Cluster"], palette="viridis", ax=ax)
    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Spending Score")
    st.pyplot(fig)