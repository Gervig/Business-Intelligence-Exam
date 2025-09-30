# Business Intelligence in the Gaming Industry: Insights on Release Timing, Platforms, and Genres
### Business Intelligence Exam for EK 4th Semester (Dat1F24)

## Context
More and more games are being developed and not all of them are successful. Companies have access to a lot of data, which includes, to name a few: Where games are being sold, release date, genres and how many were sold. The dataset that this project will be working with contains data stretching back from 1971 (CHECK THIS DATE!!!) to 2024. NÆVN AT DER ER MEGET MANGLEFULD DATA OG ANDRE TING VED DET

## Purpose
The purpose of this project is to help companies that either develop, market and or publish games to increase their sales by looking at previously released games.

## Research Questions
In this project we will investigate the following questions:

1) When should a company release a game to maximize sales at launch?

2) Do video games form distinct clusters based on release period (old vs. new) and sales performance (high vs. low)?

3) Which consoles should a specific genre game be developed for?
- ML: classficastion dependent variable: sales, independent variable: genre, console

4) How can trends be predicted so that a game is released when its genre is popular?

5) Which game genres sell better in which regions, so that companies know where to market their games? which games for which regions

6) Can we predict the sales of a game based on the different categorical values in the dataset before it launches.
- (Random Forrest)



## Hypothesis
Our hypothesis for the above mentioned questions:

1) We assume that games sell better during certain periods of the year compared to others. Therefore, we expect to observe a trend where games released in those periods achieve higher sales.

2) We expect video games can be clustered into meaningful groups according to their age and sales success

3) We assume that certain game genres sell better on specific consoles. Therefore, we expect to see a significantly larger number of sales from a specific genre on those consoles.

4) We assume that some game genres follow a popularity cycle, and we expect to identify patterns in when specific genres become popular.

5) We assume that some game genres sell better in specific regions.

6) We assume that the combination of categorical features in the dataset (such as genre, console, publisher, and developer) contains enough information to predict a game’s total sales before launch. Therefore, we expect that a machine learning model, such as a Random Forest, will be able to generate reasonably accurate predictions of sales based solely on these features.


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
-last_update = the last date that the data in the row was updated


## References
Dataset: https://mavenanalytics.io/data-playground/video-game-sales
Kaggle study: https://www.kaggle.com/datasets/asaniczka/video-game-sales-2024