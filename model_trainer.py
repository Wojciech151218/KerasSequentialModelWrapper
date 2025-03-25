

from .hyperparamaters import TrainHyperparameters


class ModelTrainer:
    def __init__(self, data_function):
        (x_train, y_train), (x_test, y_test) = data_function()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


    def get_input_shape(self):
        return self.x_train.shape[1:]


    def get_classes(self):
        return self.y_train.shape[1]


    def adapt(self, layer):
        layer.adapt(self.x_train)


    def train_model(self, model,train_hp :TrainHyperparameters,callbacks=None):
        model.compile(
            optimizer=train_hp.get_optimizer(),
            loss=train_hp.loss_function,
            metrics=train_hp.metrics,
        )
        if callbacks is None:
            callbacks = []
        model.fit(
            self.x_train,
            self.y_train,
            epochs=train_hp.epochs,
            validation_data=(self.x_test, self.y_test),
            callbacks=callbacks,
            batch_size=train_hp.batch_size
        )
