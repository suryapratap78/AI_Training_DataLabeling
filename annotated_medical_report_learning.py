# GOAL: Annotated Medical Report Data

# labeling a dataset of real or synthetic reports

data = [
    ("Chest X-ray shows infiltrates in the lower lobe. Suggestive of pneumonia.", "pneumonia"),
    ("No abnormalities seen in the chest scan. Lungs are clear.", "normal"),
    ("MRI shows a fracture in the left tibia region.", "fracture"),
    ("Scan reveals a mass in the right temporal lobe, possibly a tumor.", "tumor"),
    ("Normal chest X-ray. Heart and lung shadows are within limits.", "normal"),
]


# Train a Classifier (Text-Based)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Split the annotated data
texts, labels = zip(*data)

# Create and train a simple model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(texts, labels)

# Predict a new report
new_report = "Patient shows clear signs of infiltrates in the lungs."
prediction = model.predict([new_report])

print("Predicted condition:", prediction[0])
