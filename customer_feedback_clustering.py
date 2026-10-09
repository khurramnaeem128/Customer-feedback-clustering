
# Customer feedback clustering - NLP + K MEAN

import pandas as pd
import csv
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df = pd.read_csv(
    "sentiment-analysis.csv",
    sep=",",
    quotechar='"',
    quoting=csv.QUOTE_NONE
)

df.columns = df.columns.str.strip().str.strip('"')

df["Text"] = df["Text"].str.strip().str.strip('"')
df["Confidence Score"] = df["Confidence Score"].str.strip().str.strip('"')
print(df.head())
print(df.shape)
print(df.columns)

print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())

df = df.drop_duplicates()
print(df.shape)
print(df[df.isnull().any(axis=1)])
print(df[df["Text"].isnull()])

df = df[["Text"]]
print(df.head())
print(df.shape)

df["Text"] = df["Text"].str.lower()
print(df.head())

df["Text"] = df["Text"].str.replace(r"[^\w\s]", "", regex=True)
print(df.head())

df["Text"] = df["Text"].str.strip()
print(df.head())



nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


df["Text"] = df["Text"].apply(
    lambda text: " ".join(
        word for word in text.split()
        if word not in stop_words
    )
)

print(df.head())

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df["Text"])
print(X.shape)


inertia_values = []

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)


print(inertia_values)


k_values = range(2, 9)

plt.plot(k_values, inertia_values, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)
print(df.head())

print(df["Cluster"].value_counts().sort_index())

for cluster in sorted(df["Cluster"].unique()):
    print("\nCluster", cluster)
    print(df[df["Cluster"] == cluster]["Text"].to_string(index=False))


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())
print(X_pca.shape)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df["Cluster"],
    cmap="viridis",
    s=60
)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Customer Feedback Clusters")

plt.colorbar(label="Cluster")

plt.show()

