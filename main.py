import numpy as np
import matplotlib.pyplot as plt
import random

class NeuralNetwork:
    
    def __init__(self, layer_sizes, lr = 0.1):
        
        #tupla de tuplas que solo especifica las dimensiones de cada matriz de pesos
        weight_shapes = [(a,b) for a,b in zip(layer_sizes[1:],layer_sizes[:-1])]
        
        self.lr = lr
        
        #cantidad de capas de la NN
        self.num_layers = len(layer_sizes)
        
        #inicializar lista de matrices de pesos. /s[1]**.5 es para asegurar que cada peso tenga valores
        #cercanos a cero, asi se logra mayor pendiente en sigmoid y consecuentemente mejor entrenamiento
        self.weights = [np.random.standard_normal(s)/s[1]**.5 for s in weight_shapes]
        
        #incializar lista de matrices de biases
        self.biases = [np.zeros((j,1)) for j in layer_sizes[1:]]
    
    def predict(self, inputs):
        a = inputs
        
        #efectuar la multiplicacion matricial, es decir, FEEDFORWARD
        for w,b in zip(self.weights, self.biases):
            a = self.activation(np.dot(w,a) + b)
        return a
    
    #funcion de activacion, se eligio sigmoid
    @staticmethod
    def activation(x):
        return 1/(1 + np.exp(-x))
    
    #derivada de la funcion activacion
    def dactivation_dx(self,x):
        return self.activation(x) * (1 - self.activation(x))
    
    #calcula el error de una prediccion vs el target esperado
    def single_error(self, single_input, single_target):
        return .5 * (self.predict(single_input) - single_target) ** 2
    
    #recibe activation y no single_input asi nos ahorramos un feedforward innecesario
    def derror_dactivation(self, activation, single_target):
        return activation - single_target
    
    def _predict_each_layer(self, single_input):
        a = single_input
        activations = [a]
        zs = []
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w,a) + b
            a = self.activation(z)
            zs.append(z)
            activations.append(a)
        return tuple(zs), tuple(activations)
    
    #devuelve el gradiente del costo para cada w,b de la NN, todo para un solo elemento de training data.
    #El gradiente del costo total sera el promedio de la suma de todos los disponibles
    def single_backpropagate(self, single_input, single_target):
                
        #error = self.single_error(single_input, single_target)
        
        
        layered_zs , layered_activations = self._predict_each_layer(single_input) #feedforward
        
        
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        
        #calcular todo lo competente a la ultima capa (output) y luego efectuar backpropagation
        derror_dactivation = self.derror_dactivation(layered_activations[-1], single_target)
        delta = derror_dactivation * self.dactivation_dx(layered_zs[-1])
        nabla_w[-1] = np.dot(delta, layered_activations[-2].transpose())
        nabla_b[-1] = delta
        
        #backpropagate
        for n in range(2, self.num_layers):
            z = layered_zs[-n]
            da_dx = self.dactivation_dx(z)
            delta = np.dot(self.weights[-n+1].transpose(), delta) * da_dx
            
            nabla_w[-n] = np.dot(delta, layered_activations[-n-1].transpose())
            nabla_b[-n] = delta
            
        return nabla_w, nabla_b        
        
        
    def stochastic_gradient_descent(self, t_imgs, t_lbls, epochs = 10000):
        costos = []
        iters = []
        for i in range(epochs):
            indice_random = random.randint(0, len(t_imgs)-1)
            self.update_single(t_imgs[indice_random], t_lbls[indice_random])
            if (i % (epochs//10)) == 0:
                costo_total = sum([self.single_error(i,l) for i,l in zip(t_imgs, t_lbls)]) / len(t_imgs)
                print("Iter: {} - Costo promediado en todo el dataset: {}".format(i, sum(costo_total[0])/10))
                costos.append(costo_total[0])
                iters.append(i)
        
        return iters,costos
        
    def update_single(self, single_input, single_target):
        delta_w, delta_b = self.single_backpropagate(single_input, single_target) 
        
        self.weights = [w - self.lr * nw for w,nw in zip(self.weights, delta_w)]
        self.biases = [b - self.lr * nb for b,nb in zip(self.biases, delta_b)]
    
    def imprimir_precision(self, t_imgs, t_lbls):
        predicciones = []
        for t in t_imgs:
            predicciones.append(self.predict(t))
        num_correct = sum([np.argmax(pred) == np.argmax(lbl) for pred,lbl in zip(predicciones, t_lbls)])
        print('{}/{} precision: {}%'.format(num_correct, len(t_imgs), (num_correct/len(t_imgs))*100))
        

if __name__ == '__main__':
    
    np.random.seed(1)
    
    with np.load('mnist.npz') as data:
        training_images = data['training_images']
        training_labels = data['training_labels']
        test_img = data['test_images']
        
    layer_sizes = (784,10,10)
    
    carlos = NeuralNetwork(layer_sizes, 0.05)
    
    iters, costos = carlos.stochastic_gradient_descent(training_images, training_labels, 100000)
    plt.plot(iters,costos)
    plt.show()
    
    carlos.imprimir_precision(training_images, training_labels)
    
    while True:
        rand = random.randint(0,len(test_img) - 1)
            
        prediccion = carlos.predict(test_img[rand])
        for i,a in zip(range(layer_sizes[-1]),prediccion):
            opcional = ""
            if i == np.argmax(prediccion):
                opcional = "<--- Creo que es un {}!".format(i) 
            print("Chances de que sea {}: {}".format(i, a), opcional)
            
        plt.imshow(test_img[rand].reshape(28,28), cmap = 'gray')
        plt.show()        

            
        x = input("Comando: ")
        
        if x == 'quit':
            break


