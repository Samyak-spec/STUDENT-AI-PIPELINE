from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# 1. Define sample documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "The dog is very lazy today",
    "Cats are very different from dogs"
]

# 2. Initialize the Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# 3. Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# 4. Display as a readable Pandas DataFrame
df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
print(df)
