import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm

foo = [1, 2, 3] # krotka okrągłe nawiasy
bar = np.array(foo)

print(foo, type(foo))
print(bar, type(bar))

#########################################

foo1 = [[1, 2, 3], [4, 2, 3], [1, 2, 3]]
bar1 = np.array(foo1)
print(foo1, type(foo1))
print(bar1, type(bar1))

#########################################

# losowy zbiór danych n, zbiór cech - d

n, d = (6, 2)

X = np.random.random((n, d))
y = np.random.randint(2, size=n)  # problem binarny, ilość elementów n
print(X, X.shape)  # zawsze 2D
print(y, y.shape)  # zawsze 1D

# model rozpoznawania wzorców

np.random.seed(1410)  # ziarno

# klasyfikator

clf = NearestCentroid()
clf.fit(X, y)  # dopasowanie do modelu, zawsze X, może być X,y
print(clf.centroids_) # macierz 2x2, bo d x 2, bo binarny, jak zmienimy problem np. na 3 to będzie d x3

y_pred = clf.predict(X)
print(y, y_pred)

score = accuracy_score(y, y_pred)

print("%.3f" % score)

#########################################

n, d = (10000, 2)

X = np.random.random((n, d))
y = np.random.randint(2, size=n)

##########################################

n, d = (2000, 1000)

X, y = make_classification(n, d, d, 0, 0, n_classes=2)
np.random.seed(1410)  # ziarno

# klasyfikator
clf = NearestCentroid()

# dodane później, linijka poniżej
clf.fit(X[:1000], y[:1000])

clf.fit(X, y)  # dopasowanie do modelu, zawsze X, może być X,y
print(clf.centroids_) # macierz 2x2, bo d x 2, bo binarny, jak zmienimy problem np. na 3 to będzie d x3

# y_pred = clf.predict(X)
y_pred = clf.predict(X[100:])
print(y, y_pred)  # wektor predykcji mówi o przypisaniu do klas

# score = accuracy_score(y, y_pred)
score = accuracy_score(y[1000:], y_pred)

print("%.3f" % score)

#########################################

n, d = (2000, 1000)

X, y = make_classification(n, d, int(d/2), int(d/2), 0, n_classes=2)

print(X[:5], X.shape)
print(y[:5], y.shape)

#########################################

n, d = (2000, 1000)

X, y = make_classification(n, d, int(d/2), int(d/2), 0, n_classes=2)

pca = PCA(n_components=10)
pca.fit(X)
X_ = pca.transform(X)

print(X[:5], X.shape)
print(X_[:5], X_.shape)
print(y[:5], y.shape)

#########################################

n, d = (2000, 1000)

X, y = make_classification(n, d, int(d/2), int(d/2), 0, n_classes=2)

pca = PCA(n_components=10)
pca.fit(X)
X_ = pca.transform(X)

print(X[:5], X.shape)
print(X_[:5], X_.shape)
print(y[:5], y.shape)

clf = NearestCentroid()
clf.fit(X_[:1000], y[:1000])

print(clf.centroids_)

y_pred = clf.predict(X_[1000:])

score = accuracy_score(y[1000:], y_pred)

print("%.3f" % score)

##########################################

n, d = (2000, 1000)

X, y = make_classification(n, d, int(d/2), int(d/2), 0, n_classes=2)

for i in range(1, 41):
    pca = PCA(n_components=i)
    pca.fit(X)
    X_ = pca.transform(X)

    print(X[:5], X.shape)
    print(X_[:5], X_.shape)  # zawsze 2D
    print(y[:5], y.shape)  # zawsze 1D

    clf = NearestCentroid()
    clf.fit(X_[:1000], y[:1000])

    print(clf.centroids_)

    y_pred = clf.predict(X_[1000:])

    score = accuracy_score(y[1000:], y_pred)

    print("%i - %.3f" % (i, score))

#############################################

n, d = (2000, 1000)

X, y = make_classification(n, d, int(d/2), int(d/2), 0, n_classes=2)

accuracy_vector = np.zeros((40))

for i in range(1, 41):
    pca = PCA(n_components=i)
    pca.fit(X)
    X_ = pca.transform(X)

    clf = NearestCentroid()
    clf.fit(X_[:1000], y[:1000])

    y_pred = clf.predict(X_[1000:])

    score = accuracy_score(y[1000:], y_pred)

    print("%i - %.3f" % (i, score))
    accuracy_score[i - 1] = score

print(accuracy_score)

#############################################

n, d = (2000, 80)

X, y = make_classification(n, d, int(d/2), int(d/2), 0, n_classes=2)

accuracy_vector = np.zeros((40))

for i in tqdm(range(1, 41)):
    pca = PCA(n_components=i)
    pca.fit(X)
    X_ = pca.transform(X)

    clf = NearestCentroid()
    clf.fit(X_[:1000], y[:1000])

    y_pred = clf.predict(X_[1000:])

    score = accuracy_score(y[1000:], y_pred)

    #print("%i - %.3f" % (i, score))
    accuracy_vector[i - 1] = score

print(accuracy_vector)
plt.plot(range(1, 41), accuracy_vector)
plt.ylim(0.5, 1)
plt.savefig("foo.png")

############################################

np.random.seed(1410)

n, d, n_classes = (2000, 80, 2)

X, y = make_classification(n, d, int(d/2), int(d/2), 0, n_classes)

clf = KNeighborsClassifier()
clf.fit(X[:1000], y[:1000])

y_pred = clf.predict(X[1000:])

print(y_pred[:5], y_pred.shape)

pp = clf.predict_proba(X[1000:])
print(pp[:5], pp.shape)
