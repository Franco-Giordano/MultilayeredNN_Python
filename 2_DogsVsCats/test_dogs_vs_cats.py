import sys
sys.path.append('../src')
#asi puedo importar de src

from NeuralNetwork import NeuralNetwork
import os
import convert_image_to_mnist
import matplotlib.pyplot as plt
import numpy as np
import random

dims = (80,80)
tams = (dims[0] * dims[1],100,10,2)

np.random.seed(1)
brain = NeuralNetwork(tams, lr=0.1)

imgs_perros = []
imgs_gatos = []
imgs_perros_test = []
imgs_gatos_test = []

def convert_all_img_to_np():
    print("Comenzar a procesar imagenes de perros.")
    for root,dir,files in os.walk("../perro"):
        for img in files:
            actual = convert_image_to_mnist.import_image("./perro/" + img, dims, resize = True)
            imgs_perros.append(actual)
        print("Listo los perros")
        
    print("Comenzar a procesar imagenes de test_perro.")
    for root,dir,files in os.walk("./test_perro"):
        for img in files:
            actual = convert_image_to_mnist.import_image("./test_perro/" + img, dims, resize = True)
            imgs_perros_test.append(actual)
        print("Listo los test_perro")

    print("Comenzar a procesar las imagenes de gatos.")
    for root,dir,files in os.walk("./gato"):
        for img in files:
            actual = convert_image_to_mnist.import_image("./gato/" + img, dims, resize = True)
            imgs_gatos.append(actual)
        print("Listo los gatos")

    print("Comenzar a procesar las imagenes de test_gato.")
    for root,dir,files in os.walk("./test_gato"):
        for img in files:
            actual = convert_image_to_mnist.import_image("./test_gato/" + img, dims, resize = True)
            imgs_gatos_test.append(actual)
        print("Listo los test_gato")
    
    lbl_perros = [np.array([[1], [0]]) for u in range(len(imgs_perros))]
    lbl_gatos = [np.array([[0], [1]]) for u in range(len(imgs_gatos))]
    lbl_test_perros = [np.array([[1], [0]]) for u in range(len(imgs_perros_test))]
    lbl_test_gatos = [np.array([[0], [1]]) for u in range(len(imgs_gatos_test))]
    
    np.savez("animales.npz", gatos=imgs_gatos, perros=imgs_perros, lbl_gatos = lbl_gatos, lbl_perros = lbl_perros, test_perros = imgs_perros_test, test_gatos = imgs_gatos_test, lbl_test_perros = lbl_test_perros, lbl_test_gatos = lbl_test_gatos)

#convert_all_img_to_np()

with np.load("animales.npz") as asd:
    gatos = asd['gatos']
    perros = asd['perros']
    lbl_gatos = asd['lbl_gatos']
    lbl_perros = asd['lbl_perros']
    test_perros = asd['test_perros']
    test_gatos = asd['test_gatos']
    lbl_test_perros = asd['lbl_test_perros']
    lbl_test_gatos = asd['lbl_test_gatos']

data = np.append(gatos,perros, 0)
lbls = np.append(lbl_gatos,lbl_perros, 0)

test_data = np.append(test_gatos, test_perros, 0)
test_lbls = np.append(lbl_test_gatos, lbl_test_perros, 0)
#print(lbls.shape)

brain.load_hyperparam("hyper_dogs_cats_MEGA.npz")

es,ps = [],[]

try:
    #brain.stochastic_gradient_descent(data[:-50], lbls[:-50], 1000000)

    es, ps = brain.SGD(data, lbls, mini_batch_size=500, epochs=100, test_imgs = test_data, test_lbls = test_lbls)
    plt.plot(es,ps)
    plt.show()
    #pass
except KeyboardInterrupt:
    print("Entrenamiento detenido")
finally:
    brain.imprimir_precision(data, lbls)
    brain.imprimir_precision(test_data, test_lbls, debug = True)

    brain.save_hyperparam("hyper_dogs_cats_MEGA.npz")



posibilidades = ("perro", "gato")

"""
posibilidades = ("perro", "gato")
bien, mal = 0,0

for i in range(1000):
    rand = random.randint(0, )
    
    prediccion = brain.predict(data[rand])
    correcto = lbls[rand]
    
    
    #print(correcto)
    
    if np.argmax(prediccion) == np.argmax(correcto):
        bien += 1
    else:
        mal += 1
    
    
print("Precision en nunca vistas: {}%".format(brain.)"""
print("\n")
while True:
    rand = random.randint(-9, -1)
    
    prediccion = brain.predict(test_data[rand])
    
    for i,a in zip(range(len(posibilidades)),prediccion):
        opcional = ""
        if i == np.argmax(prediccion):
            opcional = "<------ Creo que es un {}".format(posibilidades[i])
        print("Chances de que sea {}: {} {}".format(posibilidades[i], a, opcional))

    print("\n")

    plt.imshow(test_data[rand].reshape(80,80), cmap = 'gray')
    plt.show()
    
