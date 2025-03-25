import math

from .hyperparamaters import TrainHyperparameters
from .auto_lr_finder import AutoLRFinder
import tensorflow as tf

class ModelTrainer:
    def __init__(self, data_function):
        (x_train, y_train), (x_test, y_test) = data_function()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


    def get_input_shape(self):
        return self.x_train.shape[1:]


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

    def find_learning_rate(
            self,
            model,
            train_hp :TrainHyperparameters,
            min_lr=1e-7, max_lr=1, steps=200
        ):

        lr_finder = AutoLRFinder(min_lr, max_lr, steps)
        model.compile(
            optimizer=train_hp.get_optimizer(),
            loss=train_hp.loss_function,
            metrics=train_hp.metrics,
        )
        model.fit(
            self.x_train,
            self.y_train,
            epochs=1,
            validation_data=(self.x_test, self.y_test),
            callbacks=[lr_finder],
            batch_size = 128,
            verbose= 1
        )
        lr_finder.plot()
        return lr_finder.find_optimal_lr()

