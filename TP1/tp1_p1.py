from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt 

mnist = fetch_openml('mnist_784')
images = mnist.data.reshape((-1, 28, 28))
plt.imshow(images[0],cmap=plt.cm.gray_r,interpolation="nearest")
plt.show()
