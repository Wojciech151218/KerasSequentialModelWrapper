from abc import abstractmethod

from .hyperparamaters import ArchitectureHyperparameters, TrainHyperparameters
from .model_trainer import ModelTrainer
import tensorflow as tf

class ModelBuilder:
    def __init__(self):
        self.architecture_hp : ArchitectureHyperparameters = None
        self.train_hp : TrainHyperparameters =None
        self.model_trainer : ModelTrainer = None
        self.callbacks = None

    def set_hyper_parameters(self,architecture_hp,train_hp):
        self.architecture_hp = architecture_hp
        self.train_hp = train_hp
        return self

    @staticmethod
    @abstractmethod
    def data_function(data_function = None):
        pass

    @abstractmethod
    def model_building_function(self,callbacks = None) -> tf.keras.Model:
        pass

    def get_test_set(self):
        if self.model_trainer is None:
            print("Model trainer is not set")
            return [] ,[]
        return self.model_trainer.x_test, self.model_trainer.y_test

    def set_model_trainer(self,model_trainer = None):
        if  model_trainer is None:
            self.model_trainer = ModelTrainer(self.data_function)
        else:
            self.model_trainer = model_trainer
        return self

    def set_callbacks(self,callbacks):
        self.callbacks = callbacks
        return self



    def build_model(self,model = None):
        if model is None:
            model = self.model_building_function(self.architecture_hp)

        self.model_trainer.train_model(
            model = model,
            train_hp= self.train_hp,
            callbacks = self.callbacks
        )
        return model


    def find_hp_learning_rate(
            self,
            model = None,
            min_lr=1e-7, max_lr=1, steps=100
    ):
        if model is None:
            model = self.model_building_function(self.architecture_hp)

        learning_rate = self.model_trainer.find_learning_rate(
            model = model,
            train_hp = self.train_hp,
            min_lr=min_lr, max_lr=max_lr, steps=steps
        )

        self.train_hp.learning_rate = learning_rate
        return self.train_hp
