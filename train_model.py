import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle


# Load the dataset
data=pd.read_csv("spam_email_dataset.csv", encoding='latin-1')
vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(data['email_text'])
Y=data["target"]

# Split the dataset into training and testing sets
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

# Train the KNN model
knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,Y_train)

# Predict on the test set
Y_pred=knn.predict(X_test)
# Evaluate the model
accuracy=accuracy_score(Y_test,Y_pred)
report=classification_report(Y_test,Y_pred)
conf_matrix=confusion_matrix(Y_test,Y_pred) 
print("Accuracy:",accuracy)
print("Classification Report:\n",report)
print("Confusion Matrix:\n",conf_matrix)


with open("knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)

# Save vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model saved successfully!")




