import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TrainModel:
    def __init__(self, num_layers, layer_width, batch_size, learning_rate, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = self._build_model(num_layers, layer_width)

    def _build_model(self, num_layers, layer_width):
        inputs = keras.Input(shape=(self.input_dim,))
        x = layers.Dense(layer_width, activation='relu')(inputs)
        for _ in range(num_layers):
            x = layers.Dense(layer_width, activation='relu')(x)
        outputs = layers.Dense(self.output_dim, activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='deep_q_network')
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=losses.mean_squared_error)
        return model

    def predict_single(self, state):
        state = np.reshape(state, [1, self.input_dim])
        return self.model.predict(state)

    def predict_batch(self, states):
        return self.model.predict(states)

    def train_batch(self, states, q_values):
        self.model.fit(states, q_values, epochs=1, verbose=0)

    def save_model(self, save_path):
        self.model.save(os.path.join(save_path, 'trained_model.h5'))
        plot_model(self.model, to_file=os.path.join(save_path, 'model_structure.png'),
                   show_shapes=True, show_layer_names=True)


class TestModel:
    def __init__(self, input_dim, model_path):
        self.input_dim = input_dim
        self.model = self._load_model(model_path)

    def _load_model(self, model_folder_path):
        model_file_path = os.path.join(model_folder_path, 'trained_model.h5')
        if os.path.isfile(model_file_path):
            return load_model(model_file_path)
        else:
            sys.exit("Error: Trained model file not found in the specified path.")

    def predict_single(self, state):
        state = np.reshape(state, [1, self.input_dim])
        return self.model.predict(state)
