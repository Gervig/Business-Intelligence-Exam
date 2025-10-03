import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sb
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
st.title("Business Intelligence Exam Report")

# load data
data = pd.read_csv("https://raw.githubusercontent.com/Gervig/Business-Intelligence-Exam/main/Data/vgchartz-2024.csv")

if data is not None:
    try:
        # -----------------------------
        # General Cleaning
        # -----------------------------
        st.subheader("General Cleaning")
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
        st.subheader("Q1: Best Month to Release a Game")

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
            
            st.write("""
            The Months and years shows when the game was released, not the date of the sale. Sales are measured accumulatively. But we assume that sales are highest around the release date.
            We can see a trend in later years around the month 10 and 11, which we speculate sales leading up to Christmas(the holidays). We can also see a slight trend around month 3 (March), we speculate that a lot of our data comes from games sold for NA. This could possibly be explained by holidays in NA, for example Spring break. 
            """)

            st.write("""
            We want to see how many games are released over the years for specific months, to compare with the previous animation for sales, but the sales are reported as acumelated over the years, but we figure that most of the sales come from the release period.
            Based on the 2 visualizations, we can conclude, that the months that most game are released in, thats aproximately the same months with best sales. 
            """)

            st.write("""
            ### Hypothesis for question 1: 
            We assume that games sell better during certain periods of the year compared to others. Therefore, we expect to observe a trend where games released in those periods achieve higher sales.

            ### Observartion:
            We can see that there are trends to when games are released and high sales are recorded, march, june but mostly september, october and november. We assume that is due to most of our data comes from north america, and its properly due to black friday and holidays
            """)
            

        # -----------------------------
        # Question 2
        # -----------------------------
        st.subheader("Q2: Clustering Games by Release Period & Sales")
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

            optimal_k = st.slider("Select number of clusters (K)", 2, 6, 4)
            kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
            df_q2['cluster'] = kmeans.fit_predict(X_scaled)

            st.write(f"K-Means clusters assigned ({optimal_k} clusters).")
            fig = px.scatter(
                df_q2,
                x='date_numeric',
                y='total_sales',
                color='cluster',
                labels={"date_numeric": "Days since first release", "total_sales": "Total Sales"},
                title="K-Means Clustering"
            )
            st.plotly_chart(fig)

            st.write("""
            ### Hypothesis for question 2: 
            We expect video games can be clustered into meaningful groups according to their age and sales success.

            ### Observartion:
            We can use this clustering model to help categorize different games based on their sales. We can as an example look at games in the cluster for high sales and study that type of game to see what they do differently compared to games that are placed in the lower sales cluster.
            At the same time we can potetially use older and "high" selling games to see if there is a trend that still persist in newer high selling games            
            """)

        # -----------------------------
        # Question 3
        # -----------------------------
        st.subheader("Q3: Best Console per Genre")
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

            st.write("""
            We had expected to see more sales for PC, we speculate the dataset is incomplete and doesn't reflect the real world scenario in this case. We had previously cleaned the dataset and there were thousands of rows with no data for sales, which we removed.
            """)

            st.write ("""
            ### Hypothesis for question 3:
            We assume that certain game genres sell better on specific consoles. Therefore, we expect to see a significantly larger number of sales from a specific genre on those consoles.

            ### Observartion:
            We can conclude from our data that developing games for home consoles will give the best possible sales, it is however important to note that our dataset shows clear holes in the data, with high possibility of missing data for PC releases/sales wich gives a skewed view that favors console releases.
            """)

        # -----------------------------
        # Question 4
        # -----------------------------

        st.write ("""
        ### Hypothesis for question 4:
        We assume that some game genres follow a popularity cycle, and we expect to identify patterns in when specific genres become popular.

        ### Observartion:
        We can see that aren't really any yearly trends for genres of games, we can generally see that there are some genres that are more popular than others.
        We can see that some genres used to be popular, but no longer are, such as "racing". On the other hand there are some other genres still popular, such as shooter. (note that our data cuts off at 2018). 
            """)

        # -----------------------------
        # Question 5
        # -----------------------------

        st.write ("""
        ### Hypothesis for question 5:
        We assume that some game genres sell better in specific regions.

        ### Observartion:
        We can see that the NA region has higher sells in all genres, JP and OTHER has almost none and PAL has a few. So we can conclude that all genres will most likely sell better in NA. And while Action, shooter and Sports are highest in every almost every region, it seems that JP is mostly into Role-Playing. 
            """)

        # -----------------------------
        # Question 6
        # -----------------------------
        st.subheader("Q6: Predict Sales Category with Random Forest")
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

                cm = metrics.confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                im = ax.imshow(cm, cmap=plt.cm.Blues)
                ax.set_title("Confusion Matrix")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                fig.colorbar(im)
                st.pyplot(fig)

                st.write ("""
                ### Hypothesis for question 6:
                We assume that the combination of categorical features in the dataset (such as genre, console, publisher, and developer) contains enough information to predict a game’s total sales before launch. Therefore, we expect that a machine learning model, such as a Random Forest, will be able to generate reasonably accurate predictions of sales based solely on these features.

                ### Observartion:
                Even though our model has a high prediction percentage. Most of our prediction accuracy lies in the lower bins and quickly falls off as the bins increase. That could be explain by the lower number of high selling games compared to lower selling.
 
                """)

        except Exception as e:
            st.error(f"Error in Q6: {e}")

    except Exception as e:
        st.error(f"Error processing the dataset: {e}")
else:
    st.info("Could not find data.")

# -----------------------------
# Question 7
# -----------------------------
st.subheader("Q7: Critic Score vs Total Sales with Linear Regression")

try:
    if all(x in data_clean.columns for x in ["critic_score", "total_sales"]):
        # Copy dataset with only critic_score and total_sales
        data_general_clean_date_q7 = data_clean.copy()
        data_general_clean_date_q7 = data_general_clean_date_q7.loc[:, ["critic_score", "total_sales"]]
        data_general_clean_date_q7 = data_general_clean_date_q7.dropna()

        st.subheader("Cleaned Data Sample")
        st.dataframe(data_general_clean_date_q7.head())

        # Independent (X) and dependent (y) variables
        X = data_general_clean_date_q7['critic_score'].values.reshape(-1, 1)
        y = data_general_clean_date_q7['total_sales'].values.reshape(-1, 1)

        # Train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit Linear Regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)

        a = model.coef_[0][0]   # slope
        b = model.intercept_[0] # intercept
        y_predicted = model.predict(X_test)

        # Plot Linear Regression
        st.subheader("Linear Regression Plot")
        fig, ax = plt.subplots()
        ax.set_title("Linear Regression")
        ax.scatter(X, y, color='green', label="Actual Data")
        ax.plot(X_train, a*X_train + b, color='blue', label="Regression Line")
        ax.plot(X_test, y_predicted, color='orange', linestyle="dashed", label="Predictions")
        ax.set_xlabel("Critic Score")
        ax.set_ylabel("Total Sales")
        ax.legend()
        st.pyplot(fig)

        # Show metrics
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y_test, y_predicted)
        mse = mean_squared_error(y_test, y_predicted)

        st.write(f"**R² Score:** {r2:.3f}")
        st.write(f"**Mean Squared Error:** {mse:.3f}")

        st.write("""
        ### Hypothesis for question 7:
        We assume that games with a high critic score are also the games high total sales. Therefor, we expect to see a correlation between critic score and total sales.

        ### Observartion:
        On average our prediction are off by 0.65-1.2 millions of units sold. Overall our model is a poor fit, especially for the majority of low-selling games. Our dataset has a lot of games with very low sales, so our distribution of games is heavily skewed towards that. If we wanted to make a better model we would have to normalize the total sales.
        """)

    else:
        st.info("Could not find critic_score or total_sales in dataset.")

except Exception as e:
    st.error(f"Error in Q7: {e}")
