from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy.random import randint

import time

mnist = fetch_openml('mnist_784')

# creation de l'echantillon de données a partir de a base

#for s in range(1,11):
s = 5
print("Taille de l'échantillon : "+ str(s*1000))
randIndex = randint(70000,size=s*1000)
data = mnist.data[randIndex]
target = mnist.target[randIndex]

# diviser l'echantillon de données
#for size in range(1,11):
size = 8
#print("pourcentage: "+ str(size/10))
datatrain, datatest, targettrain, targettest = train_test_split(data, target, train_size=(size/10))
# train classifier
#for n_neighbors in range(2,15):
n_neighbors = 10

start = time.time()

clf = KNeighborsClassifier (n_neighbors, p=2, n_jobs=-1)
clf.fit(datatrain, targettrain)
# prediction d'une image
'''print('predict : ' + clf.predict([datatest[4],]))
print('proba : ')
print(clf.predict_proba([datatest[4],]))
print('reponse : ' + targettest[4])'''
print('score total : ')
print(clf.score(datatest, targettest))

print ("temps d'exe: " + (str(time.time()-start)))

    #print('score appr')
    #print(clf.score(datatrain, targettrain)
