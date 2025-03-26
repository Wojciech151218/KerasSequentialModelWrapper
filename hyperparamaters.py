import tensorflow as tf

class TrainHyperparameters:
    def __init__(
            self,
            learning_rate=0.001,
            epochs=10,
            optimizer=tf.keras.optimizers.Adam,
            learning_rate_schedule=None,  # Optional learning rate harmonogram
            batch_size=32,
            loss_function="categorical_crossentropy",
            metrics=None,
            dropout_rate = 0.2,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = optimizer
        self.learning_rate_schedule = learning_rate_schedule
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.dropout_rate = dropout_rate
        self.metrics = metrics if metrics is not None else ['accuracy']

    def get_optimizer(self):
        if self.learning_rate_schedule is not None:
            # Wrap the learning rate with a schedule
            return self.optimizer(learning_rate=self.learning_rate_schedule)
        return self.optimizer(learning_rate=self.learning_rate)


class ArchitectureHyperparameters:
    def __init__(
            self,
            neurons_per_layer : [int] = None,
            neurons_per_layer_count : int = 50,
            layer_count :int = 3,
            activation_function = "relu",
            kernel_initializer = tf.keras.initializers.GlorotUniform(),
            ):
        if neurons_per_layer is None:
            self.neurons_per_layer = [neurons_per_layer_count for i in range(layer_count)]
            self.layer_count = layer_count
            self.neurons_per_layer_count = neurons_per_layer_count
        else:
            self.neurons_per_layer = neurons_per_layer
            self.layer_count = len(neurons_per_layer)
            self.neurons_per_layer_count = None
        self.activation_function = activation_function
        self.kernel_initializer = kernel_initializer