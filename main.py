import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
import time

# Streamlit UI
st.title("K-means Clustering Visualization")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("Preview of the dataset:")
    st.write(data.head())

    # Select columns for clustering
    columns = st.multiselect(
        "Select two features for clustering",
        options=data.columns,
        default=["Annual Income (k$)", "Spending Score (1-100)"]
    )

    if len(columns) == 2:
        X = data[columns]

        # Set number of clusters
        k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=5)

        # Set number of iterations
        max_iter = st.slider("Select number of iterations", min_value=1, max_value=100, value=10)


        if st.button("Run K-means Clustering"):
            st.write(f"Clustering with {k} clusters and {max_iter} iterations...")


            fig, ax = plt.subplots()
            kmeans = KMeans(n_clusters=k, init='random', max_iter=1, n_init=1)

            def update_kmeans_plot(X, centroids, labels, iteration):
                ax.clear()


                ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', s=50)


                ax.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='x')

                ax.set_title(f"K-Means Clustering - Step {iteration}")
                ax.set_xlabel(columns[0])
                ax.set_ylabel(columns[1])


                st.pyplot(fig)
                time.sleep(1)


            for i in range(1, max_iter + 1):

                kmeans = KMeans(n_clusters=k, init='random', max_iter=i, n_init=1, random_state=42)
                kmeans.fit(X)

                centroids = kmeans.cluster_centers_
                labels = kmeans.predict(X)


                update_kmeans_plot(X, centroids, labels, i)

            st.write("Clustering complete!")
