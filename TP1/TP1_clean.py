from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy.random import randint

import time

mnist = fetch_openml('mnist_784')

# ----------- programme de base ---------------------------------------
# creation de l'echantillon de données a partir de a base
s = 5
randIndex = randint(70000,size=s*1000)
data = mnist.data[randIndex]
target = mnist.target[randIndex]

# diviser l'echantillon de données
size = 8
datatrain, datatest, targettrain, targettest = train_test_split(data, target, train_size=(size/10))

# calcul du temps d'execution
start = time.time()

# train classifier
n_neighbors = 10
clf = KNeighborsClassifier (n_neighbors, p=2, n_jobs=-1) # modifier les p et n_jobs pour obtenir des resulats différents
clf.fit(datatrain, targettrain)

# prediction d'une image
print('predict : ' + clf.predict([datatest[4],]))
print('proba : ')
print(clf.predict_proba([datatest[4],]))
print('reponse : ' + targettest[4])
# calcul du score
print('score total : ')
print(clf.score(datatest, targettest))
# calcul du temps d'execution
print ("temps d'exe: " + (str(time.time()-start)))


# --------- Variation du nombre de voisins ------------------------------
# creation de l'echantillon de données a partir de a base
s = 5
randIndex = randint(70000,size=s*1000)
data = mnist.data[randIndex]
target = mnist.target[randIndex]

# diviser l'echantillon de données
size = 8
datatrain, datatest, targettrain, targettest = train_test_split(data, target, train_size=(size/10))
# train classifier
for n_neighbors in range(2,15):
	print('nombre de voisin : '+n_neighbors)
	clf = KNeighborsClassifier (n_neighbors, p=2, n_jobs=-1)
	clf.fit(datatrain, targettrain)
	# calcul du score
	print('score total : ')
	print(clf.score(datatest, targettest))

# --------- Variation du pourcentage de l'échantillon ------------------------------

# creation de l'echantillon de données a partir de a base
s = 5
randIndex = randint(70000,size=s*1000)
data = mnist.data[randIndex]
target = mnist.target[randIndex]

# diviser l'echantillon de données
for size in range(1,11):
	print("pourcentage: "+ str(size/10))
	datatrain, datatest, targettrain, targettest = train_test_split(data, target, train_size=(size/10))
	# train classifier
	n_neighbors = 10
	clf = KNeighborsClassifier (n_neighbors, p=2, n_jobs=-1)
	clf.fit(datatrain, targettrain)
	# calcul du score
	print('score total : ')
	print(clf.score(datatest, targettest))

# --------- Variation de la taille de l'echantillon ------------------------------

# creation de l'echantillon de données a partir de la base
for s in range(1,11):
	print("Taille de l'échantillon : "+ str(s*1000))
	randIndex = randint(70000,size=s*1000)
	data = mnist.data[randIndex]
	target = mnist.target[randIndex]
	# diviser l'echantillon de données
	size = 8
	datatrain, datatest, targettrain, targettest = train_test_split(data, target, train_size=(size/10))
	# train classifier
	n_neighbors = 10
	clf = KNeighborsClassifier (n_neighbors, p=2, n_jobs=-1)
	clf.fit(datatrain, targettrain)
	# calcul du score 
	print('score total : ')
	print(clf.score(datatest, targettest))


