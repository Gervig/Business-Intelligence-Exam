# Business Intelligence in the Gaming Industry: Insights on Release Timing, Platforms, and Genres
### Business Intelligence Exam for EK 4th Semester (Dat1F24)

## Context
More and more games are being developed and not all of them are successful. Companies have access to a lot of data, which includes, to name a few: Where games are being sold, release date, genres and how many were sold. The dataset that this project will be working with contains data stretching back from 1971 to 2024. 
The data we work with seems to cut off at 2018, because we removed every game that didn't have sales, generally the dataset does not reflect the real world due to the missing data from manufacturers and other factors as the impossible task of getting every downloaded games sales and so on. 


## Purpose
The purpose of this project is to help companies that either develop, market and or publish games to increase their sales by looking at previously released games.

## Research Questions
In this project we will investigate the following questions:

1) When should a company release a game to maximize sales at launch?

2) Do video games form distinct clusters based on release period (old vs. new) and sales performance (high vs. low)?

3) Which consoles should a specific genre game be developed for optimzed sales?

4) How can trends be predicted so that a game is released when its genre is popular?

5) Which game genres sell better in which regions, so that companies know where to market their games? which games for which regions

6) Can we predict the sales of a game based on the different categorical values in the dataset before it launches?

7) Does the critic score of a video games have any influence on its sales?



## Hypothesis
Our hypothesis for the above mentioned questions:

1) We assume that games sell better during certain periods of the year compared to others. Therefore, we expect to observe a trend where games released in those periods achieve higher sales.

2) We expect video games can be clustered into meaningful groups according to their age and sales success.

3) We assume that certain game genres sell better on specific consoles. Therefore, we expect to see a significantly larger number of sales from a specific genre on those consoles.

4) We assume that some game genres follow a popularity cycle, and we expect to identify patterns in when specific genres become popular.

5) We assume that some game genres sell better in specific regions.

6) We assume that the combination of categorical features in the dataset (such as genre, console, publisher, and developer) contains enough information to predict a game’s total sales before launch. Therefore, we expect that a machine learning model, such as a Random Forest, will be able to generate reasonably accurate predictions of sales based solely on these features.

7) We assume that games with a high critic score are also the games high total sales. Therefor, we expect to see a correlation between critic score and total sales.

## Observartions

1) We can see that there are trends to when games are released and high sales are recorded, March, June but mostly September, October and November. We assume that is due to most of our data comes from North America, and its properly due to black friday and holidays.

2) We can use this clustering model to help categorize different games based on their sales. We can as an example look at games in the cluster for high sales, and study that type of game to see what they do differently compared to games that are placed in the lower sales cluster.
At the same time we can potetially use older and "high" selling games to see if there is a trend that still persist in newer high selling games

3) We can conclude from our data that developing games for home consoles will give the best possible sales, it is however important to note that our dataset shows clear holes in the data, with high possibility of missing data for PC releases/sales wich gives a skewed view that favors console releases.

4) We can see that aren't really any yearly trends for genres of games, we can generally see that there are some genres that are more popular than others.
We can see that some genres used to be popular, but no longer are, such as "racing". On the other hand there are some other genres still popular, such as shooter. (note that our data cuts off at 2018). 

5) We can see that the NA region has higher sells in all genres, JP and OTHER has almost none and PAL has a few. So we can conclude that all genres will most likely sell better in NA. And while Action, shooter and Sports are highest in every almost every region, it seems that JP is mostly into Role-Playing.

6) Even though our model has a high prediction percentage. Most of our prediction accuracy lies in the lower bins and quickly falls off as the sales-bins increase. That could be explained by the lower number of high selling games compared to lower selling.

7) On average our predictions are off by 0.65-1.2 millions of units sold. Overall our model is a poor fit, especially for the majority of low-selling games. Our dataset has a lot of games with very low sales, so our distribution of games is heavily skewed towards that. There are is a trend with higher critic score and higher sales, but there is not a correlation. 

## Annotation
- This project addresses when it is most optimal to release a game in a specific genre and region.
- The success of a game is highly influenced by releasing it to the right target audience at the right time.
- Our project will be able to predict trends in the gaming market using a machine learning model.
- It will help companies avoid developing games that flop by essentially connecting the right game with the right user at the right time.

## Data units explained
- img = the uri for the box art at vgchartz.com
- title = the title
- console = the console
- genre = the genre
- publisher = the publisher
- developer = the developer
- critic_score = the metacritic score (out of 10)
- total_sales = the global sales in millions
- na_sales = the North American sales in millions
- jp_sales = the Japanese sales in millions
- pal_sales = the PAL sales in millions
- other_sales = Other sales in millions
- release_date = the release date
- last_update = the last date that the data in the row was updated


## Deployed
https://gervig-business-intelligence-exam-report-streamlit-indzmj.streamlit.app/

## References
Dataset: https://mavenanalytics.io/data-playground/video-game-sales <br>
Kaggle study: https://www.kaggle.com/datasets/asaniczka/video-game-sales-2024