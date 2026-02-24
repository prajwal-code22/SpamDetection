import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# Load the dataset
data=pd.read_csv("spam_email_dataset.csv", encoding='latin-1')
vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(data['EmailText'])
Y=data()


