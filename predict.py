import pandas as pd
import pickle

# Load model
with open("knn_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


new_email=["Congratulations! You've won a free ticket to the Bahamas. Click here to claim your prize.",
           "Dear friend, I have a business proposal for you. Please reply urgently.",
           "This is a reminder for your upcoming appointment tomorrow at 10 AM.",
           "Your order has been shipped and will be delivered by the end of the week."
                ]

email_vector = vectorizer.transform(new_email).toarray()

# Predict
prediction = model.predict(email_vector)

print("Spam" if prediction[0] == 1 else "Not Spam")