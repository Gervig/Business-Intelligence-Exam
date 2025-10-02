import pandas as pd
import numpy as np


import matplotlib.pyplot as plt

from sklearn import tree
from sklearn import model_selection


from sklearn.ensemble import RandomForestClassifier

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from yellowbrick.cluster import SilhouetteVisualizer
import sklearn.metrics as sm
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import export_text, plot_tree

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv("Data/vgchartz-2024.csv")

# We dont need the "Last_update" and the "img" column for any of our research
    
data_general_clean = data.copy()

data_general_clean = data_general_clean.drop(["last_update", "img"], axis=1)