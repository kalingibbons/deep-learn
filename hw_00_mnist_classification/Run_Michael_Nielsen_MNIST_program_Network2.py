# %%
import MNIST_Loader
import network2

training_data, validation_data, test_data = MNIST_Loader.load_data_wrapper()

net = network2.Network(sizes=[784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data,
        epochs=30,
        mini_batch_size=10,
        eta=0.5,
        lmbda=0,
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True)

# %%
