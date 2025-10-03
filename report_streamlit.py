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
        # Question 1
        # -----------------------------
        st.subheader("🎮 Q1: Best Month to Release a Game")

        if all(x in data_clean.columns for x in ["month", "year", "total_sales"]):
            df_q1 = data_clean.copy()

            # Ensure year is integer and sorted
            df_q1 = df_q1.dropna(subset=["year"])
            df_q1["year"] = df_q1["year"].astype(int)
            df_q1 = df_q1.sort_values("year")

            max_sales = df_q1.groupby(['year', 'month'])['total_sales'].sum().max()

            fig = px.histogram(
                df_q1,
                x="month",
                y="total_sales",
                animation_frame="year",
                histfunc="sum",
                category_orders={"year": sorted(df_q1["year"].unique())},  # 👈 Force chronological order
                labels={"total_sales": "Sales", "month": "Release Month", "year": "Release Year"},
                title="Monthly Release Histogram Animated by Year and Total Sales"
            )

            fig.update_yaxes(range=[0, max_sales * 1.1])
            fig.update_xaxes(range=[0.5, 12.5], tickmode='linear', dtick=1)

            st.plotly_chart(fig)
        
            # Group by year and month, count games
            games_per_month = (
                data_clean.groupby(["year", "month"])
                .size()
                .reset_index(name="count")
            )

            # Calculate max count for fixed y-axis
            max_count = games_per_month["count"].max()

            # Ensure month is integer (in case it's not)
            if games_per_month["month"].dtype == 'object':
                month_mapping = {
                    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
                }
                games_per_month["month"] = games_per_month["month"].map(month_mapping)

            games_per_month["month"] = games_per_month["month"].astype(int)

            # Create animated bar chart
            fig = px.bar(
                games_per_month,
                x="month",
                y="count",
                animation_frame="year",
                labels={
                    "count": "Number of Games",
                    "month": "Release Month",
                    "year": "Release Year"
                },
                title="Number of Games Released per Month (Animated by Year)"
            )

            # Fix both axes
            fig.update_yaxes(range=[0, max_count * 1.1])
            fig.update_xaxes(range=[0.5, 12.5], tickmode='linear', dtick=1, tick0=1)

            # Optional: sort animation frames by year
            fig.frames = sorted(fig.frames, key=lambda f: int(f.name))
            fig.layout.sliders[0]["steps"] = sorted(fig.layout.sliders[0]["steps"], key=lambda s: int(s["label"]))

            # Display in Streamlit
            st.plotly_chart(fig)

            st.write("""
            ### Hypothesis for Question 1
            We assume that games sell better during certain periods of the year compared to others. Therefore, we expect to observe a trend where games released in those periods achieve higher sales.

            ### Observation
            We can see trends in game release periods that correspond to higher sales, particularly in March, June, and mostly September, October, and November. This is likely because most of our data comes from North America, and these months coincide with Black Friday and holiday seasons.
            """)
            
        # -----------------------------
        # Question 2
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

            fig, ax = plt.subplots(figsize=(10,5))
            ax.scatter(df_q2['date'], df_q2['total_sales'], s=50, alpha=0.7)
            ax.set_xlabel("Release Date (Year-Month)")
            ax.set_ylabel("Total Sales")
            ax.set_title("Game Sales Over Time")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # -----------------------------
            # Silhouette Score Method to Find Optimal K
            # -----------------------------
            st.subheader("Silhouette Score Method for Optimal Number of Clusters")

            from sklearn.metrics import silhouette_score

            # Make sure you have the scaled clustering data (from Q2)
            # X_scaled should already be defined in your Streamlit app for clustering
            if 'X_scaled' in locals():
                K_range = range(2, 10)
                silhouette_scores = []

                for k in K_range:
                    model = KMeans(n_clusters=k, n_init=10, random_state=42)
                    labels = model.fit_predict(X_scaled)
                    score = silhouette_score(X_scaled, labels, metric='euclidean')
                    silhouette_scores.append(score)

                # Plot silhouette scores
                fig, ax = plt.subplots(figsize=(8,5))
                ax.plot(K_range, silhouette_scores, 'bx-')
                ax.set_xlabel('Number of Clusters (k)')
                ax.set_ylabel('Silhouette Score')
                ax.set_title('Silhouette Score Method for Discovering Optimal K')
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)
            else:
                st.info("Please run the clustering section first to generate X_scaled.")

        
            # K-Means clustering
            optimal_k = st.slider("Select number of clusters (K)", 2, 6, 4)
            kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
            df_q2['cluster'] = kmeans.fit_predict(X_scaled)
        
            st.write(f"K-Means clusters assigned ({optimal_k} clusters).")
            # Plot clusters
            fig = px.scatter(
                df_q2,
                x='date',
                y='total_sales',
                color='cluster',
                labels={"date": "Release Year", "total_sales": "Total Sales"},
                title="K-Means Clustering"
            )
            st.plotly_chart(fig)

        # -----------------------------
        # Question 3
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

            best_console = sales_summary.sort_values(['genre','total_sales'], ascending=[True, False]).groupby('genre').head(1)
            st.write("Best console per genre:")
            st.dataframe(best_console)

        # -----------------------------
        # Question 4
        # -----------------------------
        
        # -----------------------------
        # Total Game Sales per Year Animated by Genre
        # -----------------------------
        st.subheader("Q4: How can trends be predicted so that a game is released when its genre is popular?")

        # Group and sum total sales
        sales_summary_q4 = (
            data_clean.groupby(['year', 'genre'], as_index=False)['total_sales'].sum()
        )

        # Get all years and genres
        all_years = sales_summary_q4['year'].unique()
        all_genres = sales_summary_q4['genre'].unique()

        # Create full combination of year x genre
        full_index = pd.MultiIndex.from_product([all_years, all_genres], names=['year', 'genre'])
        sales_summary_q4 = sales_summary_q4.set_index(['year','genre']).reindex(full_index, fill_value=0).reset_index()

        # Create animated bar chart
        fig = px.bar(
            sales_summary_q4,
            x="year",
            y="total_sales",
            animation_frame="genre",
            labels={
                "total_sales": "Total Sales",
                "year": "Release Year",
                "genre": "Game Genre"
            },
            title="Total Game Sales per Year Animated by Genre",
            color="genre"  # optional, can remove if you don't want color distinction
        )

        # Fix x-axis so all years are shown
        fig.update_xaxes(categoryorder="array", categoryarray=sorted(all_years))

        # Optionally fix y-axis so bars don't get cut off
        fig.update_yaxes(range=[0, sales_summary_q4['total_sales'].max() * 1.05])

        # Display in Streamlit
        st.plotly_chart(fig)

        # -----------------------------
        # Question 5
        # -----------------------------
        st.subheader("Q5: Which game genres sell better in which regions, so that companies know where to market their games?")

        # Columns representing regions
        region_columns = ['na_sales', 'jp_sales', 'pal_sales', 'other_sales']

        # Melt into long format
        df_long = data_clean.melt(
            id_vars=['title','console','genre','publisher','developer','critic_score','total_sales','release_date'],
            value_vars=region_columns,
            var_name='region',
            value_name='sales'
        )

        # Make region names prettier
        df_long['region'] = df_long['region'].str.replace('_sales', '', regex=False).str.upper()

        # Aggregate sales by genre and region
        sales_summary_region = df_long.groupby(['genre','region'], as_index=False)['sales'].sum()

        all_genres = data_clean['genre'].unique()
        all_regions = ['NA','JP','PAL','OTHER']

        # Ensure all genre x region combinations exist
        full_index = pd.MultiIndex.from_product([all_genres, all_regions], names=['genre','region'])
        sales_summary_region = sales_summary_region.set_index(['genre','region']).reindex(full_index, fill_value=0).reset_index()

        # Create animated bar chart
        fig = px.bar(
            sales_summary_region,
            x="genre",
            y="sales",
            animation_frame="region",
            labels={
                "sales": "Total Sales",
                "genre": "Game Genre",
                "region": "Region"
            },
            title="Total Game Sales by Genre Animated by Region",
            color="genre"
        )

        # Keep x-axis fixed
        fig.update_xaxes(categoryorder="array", categoryarray=list(all_genres))

        # Fix y-axis so bars are never cut off
        fig.update_yaxes(range=[0, sales_summary_region['sales'].max() * 1.05])

        # Display in Streamlit
        st.plotly_chart(fig)

        # -----------------------------
        # Question 6
        # -----------------------------
        st.subheader("Q6: Predict Sales Category with Random Forest")

        # Raw GitHub URL
        image_url = "https://raw.githubusercontent.com/Gervig/Business-Intelligence-Exam/main/Images/random_forest.png"
        
        # Display the image
        st.image(image_url, caption="Random Forest", use_column_width=True)
        
        try:
            required_cols = ["console","genre","publisher","developer","month","year","total_sales"]
            if all(x in data_clean.columns for x in required_cols):
                df_q6 = data_clean.copy()

                drop_cols = ["title", "release_date"]
                df_q6 = df_q6.drop(columns=[c for c in drop_cols if c in df_q6.columns])

                for col in ["publisher", "developer"]:
                    topN = df_q6[col].value_counts().nlargest(20).index
                    df_q6[col] = df_q6[col].where(df_q6[col].isin(topN), "Other")

                categorical_cols = ["console", "genre", "publisher", "developer"]
                data_encoded = pd.get_dummies(df_q6, columns=categorical_cols)

                bins = [0, 0.2, 0.4, 2, 10, np.inf]
                data_encoded['sales_score_numeric'] = pd.cut(
                    data_encoded['total_sales'], bins=bins, labels=False, right=False
                )

                y = data_encoded['sales_score_numeric']
                X = data_encoded.drop(['total_sales','sales_score_numeric'], axis=1)

                X_train, X_test, y_train, y_test = model_selection.train_test_split(
                    X, y, test_size=0.15, random_state=8, stratify=y
                )

                classifier = RandomForestClassifier(
                    n_estimators=100, max_depth=6, class_weight="balanced", random_state=8
                )
                classifier.fit(X_train, y_train)

                st.success("✅ Random Forest trained on sales categories.")
                y_pred = classifier.predict(X_test)
                st.write("Accuracy:", metrics.accuracy_score(y_test, y_pred))

                # -----------------------------
                # Confusion Matrix
                # -----------------------------
                st.subheader("Confusion Matrix")

                try:
                    cm = metrics.confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    im = ax.imshow(cm, cmap=plt.cm.Blues)
                    ax.set_title("Confusion Matrix")
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    fig.colorbar(im)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"❌ Error plotting confusion matrix: {e}")

        except Exception as e:
            st.error(f"❌ Error in Q6: {e}")

    except Exception as e:
        st.error(f"❌ Error processing the dataset: {e}")
else:
    st.info("Could not find data.")
