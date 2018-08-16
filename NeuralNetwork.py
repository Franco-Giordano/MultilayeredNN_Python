import numpy as np
import matplotlib.pyplot as plt
import random
import convert_image_to_mnist
import save_painting

class NeuralNetwork:
    
    def __init__(self, layer_sizes, lr = 0.1):
        
        #tupla de tuplas que solo especifica las dimensiones de cada matriz de pesos
        weight_shapes = [(a,b) for a,b in zip(layer_sizes[1:],layer_sizes[:-1])]
        
        self.lr = lr
        
        self.layer_sizes = layer_sizes
        
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
    
    def obtener_precision(self, t_imgs, t_lbls, extras = False):
 
        predicciones = []
        for t in t_imgs:
            predicciones.append(self.predict(t))
        num_correct = sum([np.argmax(pred) == np.argmax(lbl) for pred,lbl in zip(predicciones, t_lbls)])
        
        if extras:
            return num_correct, len(t_imgs), (num_correct/len(t_imgs))*100
        return (num_correct/len(t_imgs))*100
    
    def imprimir_precision(self, t_imgs, t_lbls, debug = False):
        
        num_correct, total, porcentaje = self.obtener_precision(t_imgs, t_lbls, extras = True)
        
        print('{}/{} precision: {}%'.format(num_correct, total, porcentaje))
        
        if debug:
            print("Ajustes:\n    LR: {}\n    Capas: {}".format(self.lr, self.layer_sizes))
    
    def save_hyperparam(self, path = "hyperparameters.npz"):
        np.savez(path, weights = self.weights, biases = self.biases, layer_sizes = self.layer_sizes)

    def load_hyperparam(self, path):
        
        with np.load(path) as data:
            self.weights = data['weights']
            self.biases = data['biases']
            self.layer_sizes = tuple(data['layer_sizes'])
            
    def SGD(self, t_imgs, t_lbls, mini_batch_size=100, epochs = 40, test_imgs = None, test_lbls = None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        random.seed(1)
        if test_imgs and test_lbls: 
            n_test = len(test_imgs)
        n = len(t_imgs)
        for j in range(epochs):
            s = np.arange(t_imgs.shape[0])
            np.random.shuffle(s)
            t_imgs = t_imgs[s]
            t_lbls = t_lbls[s]            
            mini_batches = [(t_imgs[k:k+mini_batch_size],t_lbls[k:k+mini_batch_size]) for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch[0], mini_batch[1])
            print("Epoch {} complete".format(j))

    def update_mini_batch(self, mb_imgs, mb_lbls):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in zip(mb_imgs, mb_lbls):
            delta_nabla_w, delta_nabla_b = self.single_backpropagate(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(self.lr/len(mb_imgs))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(self.lr/len(mb_imgs))*nb for b, nb in zip(self.biases, nabla_b)]


if __name__ == '__main__':
    
    np.random.seed(1)
    
    with np.load('mnist.npz') as data:
        training_images = data['training_images']
        training_labels = data['training_labels']
        test_img = data['test_images']
        
    layer_sizes = (784,10,10)
    
    carlos = NeuralNetwork(layer_sizes, 0.1)
    
    """iters, costos = carlos.stochastic_gradient_descent(training_images, training_labels, 100000)
    plt.plot(iters,costos)
    plt.show()"""
    
    carlos.SGD(training_images, training_labels, 50)
    
    carlos.imprimir_precision(training_images, training_labels)
    
    carlos.save_hyperparam("hyper_mnist_sgd_batches.npz")
    
    
    while True:
        
        painter = save_painting.Painter(200, 200, (0,0,0))
        painter.create_canvas()
        
        array_dibujado = convert_image_to_mnist.import_image("image.png", (28,28))
            
        prediccion = carlos.predict(array_dibujado)
        
        for i,a in zip(range(layer_sizes[-1]),prediccion):
            opcional = ""
            if i == np.argmax(prediccion):
                opcional = "<--- Creo que es un {}!".format(i)
            print("Chances de que sea {}: {}".format(i, a), opcional)
            
        plt.imshow(array_dibujado.reshape(28,28), cmap = 'gray')
        plt.show()
        

            
        x = input("Comando: ")
        
        if x == 'quit':
            break


