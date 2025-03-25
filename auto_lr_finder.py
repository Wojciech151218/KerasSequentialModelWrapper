import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class AutoLRFinder(tf.keras.callbacks.Callback):
    def __init__(self, min_lr=1e-7, max_lr=1, steps=100):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.steps = steps
        self.lr_mult = (max_lr / min_lr) ** (1 / steps)
        self.lrs = []
        self.losses = []
        self.best_loss = np.inf
        self.best_lr = None

    def on_train_batch_begin(self, batch, logs=None):
        lr = self.min_lr * (self.lr_mult ** len(self.lrs))
        self.model.optimizer.learning_rate = lr
        self.lrs.append(lr)

    def on_train_batch_end(self, batch, logs=None):
        loss = logs["loss"]
        self.losses.append(loss)

        # Stop if loss explodes
        if loss < self.best_loss:
            self.best_loss = loss
        elif loss > 4 * self.best_loss:
            self.model.stop_training = True  # Stop training if loss explodes

    def find_optimal_lr(self):
        """ Automatically finds the best learning rate based on the steepest loss drop """
        loss_arr = np.array(self.losses)
        lr_arr = np.array(self.lrs)

        #loss_gradient = np.gradient(loss_arr)
        #min_grad_idx = np.argmin(loss_gradient)

        smallest_loss_idx = np.argmin(loss_arr)

        # Choose a slightly lower learning rate for stability
        self.best_lr = lr_arr[smallest_loss_idx] / 10  # Picking 10x lower than the steepest drop

        return self.best_lr

    def plot(self):
        plt.figure(figsize=(8, 6))
        plt.semilogx(self.lrs, self.losses, label="Loss")
        if self.best_lr:
            plt.axvline(self.best_lr, color='r', linestyle='--', label=f"Best LR: {self.best_lr:.2e}")
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")
        plt.legend()
        plt.show()