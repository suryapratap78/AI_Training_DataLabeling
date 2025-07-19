# Goal: Automatically classify test logs as “pass”, “fail”, or “flaky”.

logs = [
    ("Test passed in 5.2s with no errors", "pass"),
    ("Timeout error at line 32", "fail"),
    ("Test failed once but passed on retry", "flaky"),
    ("All tests completed successfully", "pass"),
]

# Model Training:

texts, labels = zip(*logs)
X = CountVectorizer().fit_transform(texts)
model = MultinomialNB().fit(X, labels)

print(model.predict([CountVectorizer().fit_transform(["Error occurred during setup"]).toarray()[0]]))

