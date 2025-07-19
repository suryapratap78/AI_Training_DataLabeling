# example of data annotation and training an AI using a text classification use case — like training a model 
# to detect whether a movie review is positive or negative.

# Labeled (annotated) data
data = [
    ("This movie was fantastic!", "positive"),
    ("I hated the film. It was boring.", "negative"),
    ("Absolutely loved the acting and story.", "positive"),
    ("Terrible plot and weak characters.", "negative"),
]



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Step 1: Split data
texts, labels = zip(*data)

# Step 2: Convert text to numeric features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Step 3: Train the model
model = MultinomialNB()
model.fit(X, labels)

# Step 4: Make a prediction
test_text = ["The movie was amazing and emotional."]
X_test = vectorizer.transform(test_text)
prediction = model.predict(X_test)

print("Prediction:", prediction[0])
