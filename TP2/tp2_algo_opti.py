from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, zero_one_loss
from numpy import concatenate
import time

# importation des donnees
mnist = fetch_openml('mnist_784')

# division de la base : 49000 lignes training 
datatrain, datatest, targettrain, targettest = train_test_split(mnist.data, mnist.target, train_size=25000)

sizes = [2,10,20,30,40,50,100]

# variation des algo d'opti

algos = ['lbfgs','sgd','adam']


# variation du nombre de couches de 50
for j in sizes:
    for i in algos:
        print('nb de couches: '+ str(j))
        print('Algo d opti: ' + i)
        array = [50 for x in range(j)] 
        clf = MLPClassifier(hidden_layer_sizes=tuple(array), solver=i)

        start = time.time()
        # entrainement
        clf.fit(datatrain, targettrain)
        print ("temps d'exe: " + (str(time.time()-start)))

        # calcul de la precision
        yprediction = clf.predict(datatest)
        print("precision : "+ str(precision_score(targettest, yprediction,average='micro')))

        # calcul de l'erreur
        print("erreur : " + str(zero_one_loss(targettest, yprediction)))
