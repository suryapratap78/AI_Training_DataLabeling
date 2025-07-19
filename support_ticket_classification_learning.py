# Goal: Automatically classify support tickets into categories.

data = [
    ("My order hasn't arrived yet.", "delivery_issue"),
    ("I want to change my password.", "account_help"),
    ("Product was damaged on arrival.", "product_issue"),
    ("I can't log into my account.", "account_help"),
]


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts, labels = zip(*data)
X = CountVectorizer().fit_transform(texts)
model = MultinomialNB().fit(X, labels)

print(model.predict(X[-1]))  # Predicts category for last ticket
