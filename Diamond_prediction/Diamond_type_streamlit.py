import pickle
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- Load models ----------------
with open('Random Forest Regressor_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('clustering_kmeans.pkl', 'rb') as f:
    cluster = pickle.load(f)

# ---------------- Load dataset ----------------
df_clustered = pd.read_csv("clustered_data.csv")
cluster_summary = df_clustered.groupby('Cluster')[['Price per Carat','Volume']].mean().round(2)

# ---------------- Sidebar navigation ----------------
page = st.sidebar.radio(
    "Navigation",
    ["Diamond Price Prediction", "Clustering Analysis", "Diamond Cluster Classifier", "Cluster Visualization"]
)

# ---------------- Encoding dictionaries ----------------
color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
Carat_Category_order = ['Light', 'Medium', 'Heavy']
Cluster_labels = ['0','1','2']
Segment_labels = ['Very Small Diamonds','Mid-range Diamonds','Premium Heavy Diamonds']

color_map = {cat: idx for idx, cat in enumerate(color_order)}
cut_map = {cat: idx for idx, cat in enumerate(cut_order)}
clarity_map = {cat: idx for idx, cat in enumerate(clarity_order)}
carat_cat_map = {cat: idx for idx, cat in enumerate(Carat_Category_order)}

# ---------------- Diamond Price Prediction ----------------
if page == "Diamond Price Prediction":
    st.title("💎 Diamond Price Prediction")

    # Inputs
    x = st.number_input("x", min_value=1.0, max_value=10.0, value=1.0)
    y = st.number_input("y", min_value=1.0, max_value=10.0, value=1.0)
    z = st.number_input("z", min_value=1.0, max_value=10.0, value=1.0)
    Volume = st.number_input("Volume", min_value=4.0, max_value=500.0, value=4.0)
    Price_per_Carat = st.number_input("Price_per_Carat", min_value=1, max_value=20000, value=1)
    Dimension_Ratio = st.number_input("Dimension_Ratio", min_value=0.0, max_value=4.0, value=0.0)
    color = st.selectbox("Color", color_order)
    cut = st.selectbox("Cut", cut_order)
    clarity = st.selectbox("Clarity", clarity_order)
    carat = st.selectbox("Carat Category", Carat_Category_order)

    if st.button("Predict Price"):
        # Encode categorical features
        color_encoded = color_map[color]
        cut_encoded = cut_map[cut]
        clarity_encoded = clarity_map[clarity]
        carat_cat_encoded = carat_cat_map[carat]

        # Create input array (order must match training features!)
        input_data = [[x, y, z, Volume, Price_per_Carat, Dimension_Ratio,
                       color_encoded, cut_encoded, clarity_encoded, carat_cat_encoded]]

        # Predict
        price_log = model.predict(input_data)[0]  # model predicts log(price)
        # price = np.expm1(price_log)  # if model trained on log(price)

        st.success(f"💎 Predicted Diamond Price in Dollars: ${price_log:,.2f}")
        st.success(f"💎 Predicted Diamond Price in INR: ₹{price_log*83:,.2f}")

# ---------------- Clustering Analysis ----------------
elif page == "Clustering Analysis":
    st.title("📊 Clustering Analysis")
    st.dataframe(cluster_summary)

# ---------------- Diamond Cluster Classifier ----------------
elif page == "Diamond Cluster Classifier":
    st.title("💎 Diamond Cluster Classifier")

    price = st.slider("Price (INR)", min_value=1000, max_value=55000, value=1000)
    carat = st.selectbox("Carat Category", Carat_Category_order)
    carat_cat_encoded = carat_cat_map[carat]

    if st.button("Classify Diamond"):
        df_features = df_clustered[['Carat_Cat_encoded', 'Price per Carat', 'Cluster', 'Segment']].copy()
        distances = ((df_features['Carat_Cat_encoded'] - carat_cat_encoded) ** 2 +
                     (df_features['Price per Carat'] - price) ** 2)
        closest_idx = distances.idxmin()
        cluster_name = df_features.loc[closest_idx, 'Segment']
        st.success(f"This diamond belongs to **{cluster_name}** cluster.")

# ---------------- Cluster Visualization ----------------
elif page == "Cluster Visualization":
    st.header("📈 Cluster Visualization")
    st.write("Scatter plot of Volume vs Price per Carat, Segment")

    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(
        data=df_clustered,
        x="Volume", y="Price per Carat",
        hue="Segment",
        palette="Set2",
        ax=ax
    )
    ax.set_title("Diamond Clusters")
    st.pyplot(fig)

