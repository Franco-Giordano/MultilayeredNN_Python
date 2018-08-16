from NeuralNetwork import NeuralNetwork
import os
import convert_image_to_mnist
import matplotlib.pyplot as plt
import numpy as np
import random

dims = (80,80)
tams = (dims[0] * dims[1],10,10,2)

np.random.seed(10)

brain = NeuralNetwork(tams)

imgs_perros = []
imgs_gatos = []

def convert_all_img_to_np():
    for root,dir,files in os.walk("./perro"):
        for img in files:
            actual = convert_image_to_mnist.import_image("./perro/" + img, dims, resize = True)
            imgs_perros.append(actual)
        print("Listo los perros")
            
    for root,dir,files in os.walk("./gato"):
        for img in files:
            actual = convert_image_to_mnist.import_image("./gato/" + img, dims, resize = True)
            imgs_gatos.append(actual)
        print("Listo los gatos")
    
    lbl_perros = [np.array([[1], [0]]) for u in range(len(imgs_perros))]
    lbl_gatos = [np.array([[0], [1]]) for u in range(len(imgs_gatos))]
    
    np.savez("animales.npz", gatos=imgs_gatos, perros=imgs_perros, lbl_gatos = lbl_gatos, lbl_perros = lbl_perros)

#convert_all_img_to_np()

with np.load("animales.npz") as asd:
    gatos = asd['gatos']
    perros = asd['perros']
    lbl_gatos = asd['lbl_gatos']
    lbl_perros = asd['lbl_perros']

data = np.append(gatos,perros, 0)
lbls = np.append(lbl_gatos,lbl_perros, 0)
#print(lbls.shape)

#brain.load_hyperparam("hyper_dogs_cats.npz")

#brain.stochastic_gradient_descent(data[:-50], lbls[:-50], 1000000)

brain.SGD(data[:-50], lbls[:-50], mini_batch_size=70, epochs=1000)

brain.imprimir_precision(data, lbls, debug = True)

#brain.save_hyperparam("hyper_dogs_cats.npz")

posibilidades = ("perro", "gato")
bien, mal = 0,0

for i in range(500):    
    rand = random.randint(-50, -1)
    
    prediccion = brain.predict(data[rand])
    correcto = lbls[rand]
    
    
    print(correcto)
    
    if np.argmax(prediccion) == np.argmax(correcto):
        bien += 1
    else:
        mal += 1
    
    for i,a in zip(posibilidades,prediccion):
        print("Chances de que sea {}: {}".format(i, a))
    
    print("Precision hasta ahora: {}".format((bien/(bien+mal))*100))
    print("\n")
    #plt.imshow(data[rand].reshape(80,80), cmap = 'gray')
    #plt.show()
    
    
