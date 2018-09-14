import sys
sys.path.append('../src')
# asi puedo importar de src

from NeuralNetwork import NeuralNetwork
import random
import numpy as np
import matplotlib.pyplot as plt

def main():

    data = np.load('../1_MNIST/mnist.npz')

    configsPosibles = {'num_inputs': 784, 'num_outputs': 10, 'max_num_layers': 2, 'max_num_neuronas': 10}

    tamPoblacion = 4

    numGeneraciones = 6

    mutador = Mutador(tamPoblacion, numGeneraciones, configsPosibles, data)

    mutador.crearPoblacionInicial()

    optima = mutador.correrGeneraciones()

    gens = list(range(numGeneraciones+1))

    print("Mejor precision: ", optima[1])

    optima[0].save_hyperparam(path="genetic_hyper.npz")

    plt.plot(gens, mutador.fitnessPromedioPorGen)
    plt.show()

class Mutador:

    def __init__(self, tamPoblacion, numGeneraciones, configsPosibles, dataset):

        self.tamPoblacion = tamPoblacion
        self.numGeneraciones = numGeneraciones
        self.configsPosibles = configsPosibles
        self.dataset = dataset
        self.poblacionCalificada = []
        self.fitnessPromedioPorGen = []

    def crearPoblacionInicial(self):

        print("CREANDO POBLACION INICIAL-----------------------")
        for _ in range(self.tamPoblacion):

            cantLayers = random.randint(1, self.configsPosibles['max_num_layers'])

            tam_capas = [random.randint(1, self.configsPosibles['max_num_neuronas']) for _ in range(cantLayers)]
            tam_capas.insert(0, self.configsPosibles['num_inputs'])
            tam_capas.append(self.configsPosibles['num_outputs'])

            nn = NeuralNetwork(tam_capas)

            self.poblacionCalificada.append([nn, 0])

        self.entrenarYCalificarPoblacion()

    def correrGeneraciones(self):

        for g in range(self.numGeneraciones):
            print("INICIO GENERACION {}-----------------------".format(g))
            self.evolucionarPoblacion()
            self.entrenarYCalificarPoblacion()
            print("     PROMEDIO FITNESS: {}".format(self.fitnessPromedioPorGen[-1]))

        mejor = self.getMejorHabitanteActual()

        return mejor

    def entrenarYCalificarPoblacion(self):

        print("     Iniciando entrenamiento de poblacion...")
        forma = [p[0].layer_sizes for p in self.poblacionCalificada]
        print("         Forma poblacion: {}".format(forma))
        for i, tupla in enumerate(self.poblacionCalificada):
            nn = tupla[0]

            nn.SGD(self.dataset['training_images'], self.dataset['training_labels'], epochs=10)  # TODO: HACER QUE NO IMPRIMA EL SGD -----------------------------------------------

            tupla[1] = self.evaluarFitness(nn)

            if i%2 == 0:
                print("         Entrenadas {}/{}".format(i+1, self.tamPoblacion))

        self.fitnessPromedioPorGen.append(self.calcularPromedioFitness())

    def calcularPromedioFitness(self):
        total = sum([t[1] for t in self.poblacionCalificada])
        return total / self.tamPoblacion

    def evaluarFitness(self, neuralNetwork):

        precision = neuralNetwork.obtener_precision(self.dataset['test_images'], self.dataset['test_labels'])
        return precision

    def evolucionarPoblacion(self):

        mejores, randoms = self.filtrarMejoresYRandom(porcentajeMejores=0.3, porcentajeRndm=0.3)

        pobParcial = mejores + randoms

        hijosFaltantes = self.tamPoblacion - len(pobParcial)

        hijos = []

        print("     Criando {} hijos...".format(hijosFaltantes))
        for h in range(hijosFaltantes):
            padre = pobParcial[random.randint(0, len(pobParcial) - 1)]
            madre = pobParcial[random.randint(0, len(pobParcial) - 1)]
            i=0
            while padre == madre and i < 10:
                madre = pobParcial[random.randint(0, len(pobParcial) - 1)]
                i+=1

            hijo = self.criar(padre, madre)

            hijos.append(hijo)

        print("     Mutando padres...")
        self.mutarUnaPoblacion(pobParcial)

        pobParcial.extend(hijos)

        random.shuffle(pobParcial)

        assert len(pobParcial) == self.tamPoblacion

        self.poblacionCalificada = [[nn, 0] for nn in pobParcial]

    def getMejorHabitanteActual(self):
        return max(self.poblacionCalificada, key=lambda x: x[1])

    def filtrarMejoresYRandom(self, porcentajeMejores=0.3, porcentajeRndm=0.1):
        cantMejores = round(self.tamPoblacion * porcentajeMejores)
        cantRndm = round(self.tamPoblacion * porcentajeRndm)

        pobOrdenada = sorted(self.poblacionCalificada, key= lambda x: x[1])
        pobOrdenada = [x[0] for x in pobOrdenada]  # sacar valores de fitness

        mejores = pobOrdenada[:cantMejores]
        rndms = pobOrdenada[cantMejores:]

        np.random.shuffle(rndms)
        rndms = rndms[:cantRndm]

        return mejores, rndms

    def criar(self, padre, madre):
        propPadre = padre.layer_sizes[1:-1]
        propMadre = madre.layer_sizes[1:-1]  # ignoro capas input y output

        propHijo = []

        for i in range(min(len(propPadre), len(propMadre))):
            propHijo.insert(i, random.choice((propPadre[i], propMadre[i])))

        masGrande = max((propPadre, propMadre), key=lambda x: len(x))

        for i in range(len(propHijo), len(masGrande)):
            agrandar = random.randint(0, 1)

            if agrandar:
                propHijo.append(masGrande[i])

        propHijo = [self.configsPosibles['num_inputs']] + propHijo + [self.configsPosibles['num_outputs']]

        return NeuralNetwork(propHijo)

    def mutarUnaPoblacion(self, pob):

        for i in range(len(pob)):
            propsNN = pob[i].layer_sizes

            capaRandom = random.randint(1, len(propsNN) - 2)
            propsNN[capaRandom] = random.randint(1, self.configsPosibles['max_num_neuronas'])

            pob[i] = NeuralNetwork(propsNN)

if __name__ == '__main__':
    main()
