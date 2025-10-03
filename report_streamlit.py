import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import model_selection, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.tree import plot_tree

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Business Intelligence Report", layout="wide")
st.title("📊 Business Intelligence Exam Report")

# load data
data = pd.read_csv("https://raw.githubusercontent.com/Gervig/Business-Intelligence-Exam/main/Data/vgchartz-2024.csv")

if data is not None:
    try:
        # -----------------------------
        # General Cleaning
        # -----------------------------
        st.subheader("🧹 General Cleaning")
        data_clean = data.copy()
        # Drop unneeded columns if they exist
        for col in ["last_update", "img"]:
            if col in data_clean.columns:
                data_clean.drop(col, axis=1, inplace=True)

        # Drop rows without total_sales
        if "total_sales" in data_clean.columns:
            data_clean.dropna(subset=["total_sales"], inplace=True)

        # Fill NA for sales regions with 0
        for col in ["na_sales", "jp_sales", "pal_sales", "other_sales"]:
            if col in data_clean.columns:
                data_clean[col] = data_clean[col].fillna(0)

        st.write("Data after general cleaning:")
        st.dataframe(data_clean.head())

        # -----------------------------
        # Extract Month/Year from release_date
        # -----------------------------
        if "release_date" in data_clean.columns:
            data_clean["release_date"] = pd.to_datetime(data_clean["release_date"], errors='coerce')
            data_clean = data_clean.dropna(subset=["release_date"])
            data_clean["month"] = data_clean["release_date"].dt.month.astype("Int64")
            data_clean["year"] = data_clean["release_date"].dt.year.astype("Int64")

        # -----------------------------
        # Question 1: Best month to release a game
        # -----------------------------
        st.subheader("🎮 Q1: Best Month to Release a Game")

        if all(x in data_clean.columns for x in ["month", "year", "total_sales"]):
            df_q1 = data_clean.copy()

            max_sales = df_q1.groupby(['year', 'month'])['total_sales'].sum().max()
            fig = px.histogram(
                df_q1,
                x="month",
                y="total_sales",
                animation_frame="year",
                histfunc="sum",
                labels={"total_sales": "Sales", "month": "Release Month", "year": "Release Year"},
                title="Monthly Release Histogram Animated by Year and Total Sales"
            )
            fig.update_yaxes(range=[0, max_sales * 1.1])
            fig.update_xaxes(range=[0.5, 12.5], tickmode='linear', dtick=1)
            st.plotly_chart(fig)

        # -----------------------------
        # Question 2: Clustering based on release period & sales
        # -----------------------------
        st.subheader("📊 Q2: Clustering Games by Release Period & Sales")

        if all(x in data_clean.columns for x in ["year", "month", "total_sales"]):
            df_q2 = data_clean.copy()
            df_q2 = df_q2.dropna(subset=["year", "month"])
            df_q2['date'] = pd.to_datetime(dict(year=df_q2["year"], month=df_q2["month"], day=1))
            df_q2['date_numeric'] = (df_q2['date'] - df_q2['date'].min()).dt.days
            X_clustering = df_q2[['date_numeric', 'total_sales']].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clustering)

            st.write("Scatter plot of sales over time:")
            fig, ax = plt.subplots(figsize=(10,5))
            ax.scatter(df_q2['date'], df_q2['total_sales'], s=50, alpha=0.7)
            ax.set_xlabel("Release Date (Year-Month)")
            ax.set_ylabel("Total Sales")
            ax.set_title("Game Sales Over Time")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Determine number of clusters
            optimal_k = st.slider("Select number of clusters (K)", 2, 6, 4)
            kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
            df_q2['cluster'] = kmeans.fit_predict(X_scaled)

            st.write(f"K-Means clusters assigned ({optimal_k} clusters).")

            st.write("Cluster Visualization:")
            fig = px.scatter(
                df_q2,
                x='date_numeric',
                y='total_sales',
                color='cluster',
                labels={"date_numeric": "Days since first release", "total_sales": "Total Sales"},
                title="K-Means Clustering"
            )
            st.plotly_chart(fig)

            # SilhouetteVisualizer
            # st.write("Silhouette Visualizer:")
            # fig, ax = plt.subplots()
            # visualizer = SilhouetteVisualizer(KMeans(n_clusters=optimal_k), ax=ax)
            # visualizer.fit(X_scaled)
            # st.pyplot(fig)

        # -----------------------------
        # Question 3: Best console per genre
        # -----------------------------
        st.subheader("🎮 Q3: Best Console per Genre")
        if all(x in data_clean.columns for x in ["genre", "console", "total_sales"]):
            df_q3 = data_clean.copy()
            all_consoles = df_q3['console'].unique()
            all_genres = df_q3['genre'].unique()
            full_index = pd.MultiIndex.from_product([all_genres, all_consoles], names=['genre', 'console'])
            sales_summary = df_q3.groupby(['genre', 'console'])['total_sales'].sum().reindex(full_index, fill_value=0).reset_index()

            fig = px.bar(
                sales_summary,
                x="console",
                y="total_sales",
                animation_frame="genre",
                labels={"total_sales": "Total Sales", "console": "Console", "genre": "Genre"},
                title="Sales per Console Animated by Genre"
            )
            fig.update_xaxes(categoryorder="array", categoryarray=list(all_consoles))
            st.plotly_chart(fig)

            # Show top console per genre
            best_console = sales_summary.sort_values(['genre','total_sales'], ascending=[True, False]).groupby('genre').head(1)
            st.write("Best console per genre:")
            st.dataframe(best_console)

        # -----------------------------
        # Question 6: Random Forest Sales Prediction Example
        # -----------------------------
        st.subheader("🧠 Q6: Predict Sales Category with Random Forest")

        if all(x in data_clean.columns for x in ["console","genre","publisher","developer","month","year","total_sales"]):
            df_q6 = data_clean.copy()
            # Drop columns that shouldn't be used as features (like title)
            drop_cols = ["title", "release_date"]  # add more if needed
            df_q6 = df_q6.drop(columns=[c for c in drop_cols if c in df_q6.columns])
            categorical_cols = ["console", "genre", "publisher", "developer"]
            data_encoded = pd.get_dummies(df_q6, columns=categorical_cols).astype(float)
            bins = [0, 0.2, 0.4, 2, 10, np.inf]
            labels = ['0.2', '0.2-0.4', '0.4-2', '2-10', "10+"]
            data_encoded['sales_score_numeric'] = pd.cut(data_encoded['total_sales'], bins=bins, labels=False, right=False)
            y = data_encoded['sales_score_numeric']
            X = data_encoded.drop(['total_sales','sales_score_numeric'], axis=1)

            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.15, random_state=8)
            classifier = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=8)
            classifier.fit(X_train, y_train)

            st.write("Random Forest trained on sales categories.")
            y_pred = classifier.predict(X_test)
            st.write("Accuracy:", metrics.accuracy_score(y_test, y_pred))

            st.write("Confusion Matrix:")
            cm = metrics.confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            ax.imshow(cm, cmap=plt.cm.Blues)
            ax.set_title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error processing the dataset: {e}")

else:
    st.info("👆 Upload a CSV or Excel dataset to get started")
