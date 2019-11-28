from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, zero_one_loss, recall_score
from sklearn.metrics import confusion_matrix
from numpy.random import randint

import time

# importation des donnees
mnist = fetch_openml('mnist_784')

randIndex = randint(70000,size=10000)
data = mnist.data[randIndex]
target = mnist.target[randIndex]

# division de la base : 49000 lignes training 
datatrain, datatest, targettrain, targettest = train_test_split(data, target, train_size=0.7)

c = [0.1,0.25,0.5,0.75,1]

for i in c:

    print('parametre C : ' + str(i))    

    # construction du modele
    clf = SVC(C=i, gamma='scale')

    start = time.time()
    # entrainement
    clf.fit(datatrain, targettrain)
    print ("temps d'exe: " + (str(time.time()-start)))

    # prediction sur les donnees test
    yprediction = clf.predict(datatest)
    # calcul de la precision
    print("precision : "+ str(precision_score(targettest, yprediction,average='micro')))

    # calcul de l'erreur sur les données de test
    print("erreur : " + str(zero_one_loss(targettest, yprediction)))

    # calcul du recall
    print("recall : " + str(recall_score(targettest, yprediction, average='macro')))
# average=none pour avoir le detail par chiffre

    # construction de la matrice de confusion
    cm = confusion_matrix(targettest, yprediction) 
    print('matrice de confusion : ' + str(cm))


    # prediction sur les donnees d'entrainement
    ypredictiontrain = clf.predict(datatrain)

    # calcul de l'erreur sur les données d'entrainement
    print("erreur training : " + str(zero_one_loss(targettrain, ypredictiontrain)))


