from NeuralNetwork import *

np.random.seed(1)
    
with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']
    test_img = data['test_images']

lrs = (0.4,0.3,0.2, 0.1, 0.05)
results = []
i =1
for lr in lrs:
    print("--------------ENTRENAMIENTO {} DE {} CON {}--------------".format(i,len(lrs),lr))
    layer_sizes = (784,10,10)
        
    carlos = NeuralNetwork(layer_sizes, lr)
        
    iters, costos = carlos.stochastic_gradient_descent(training_images, training_labels, 100000)
    plt.plot(iters,costos)
        
    results.append(carlos.obtener_precision(training_images, training_labels))
    
    i += 1
    
    
for r,lr in zip(results,lrs):
    print("Con {} precision de {}%".format(lr, r))

axes = plt.gca()
axes.set_xlim([30000,90000])
axes.set_ylim([0, 0.007])
plt.show()