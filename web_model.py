#very sad hotfix
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import plot_model
from typing import *



class vgg():
    model = None
    checkpoint = None
    tensorboard = None
    train_path = None
    logpath = None

    def __init__(self, train_path: str, logpath: str, modelpath: str, h5_file: str=None):
        self.train_path = train_path
        self.logpath = logpath
        self.modelpath = modelpath
        self.checkpoint = self.get_checkpoint(modelpath)
        self.tensorboard = self.get_tensorboard(logpath)
        self.model = self.get_model(h5_file)

    def get_checkpoint(self, modelpath: str) -> ModelCheckpoint:
        return ModelCheckpoint(
            modelpath,
            monitor="loss",
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
            options=None
        )

    def get_tensorboard(self, logpath) -> TensorBoard:
        return TensorBoard(
            logpath,
            histogram_freq=0, 
            write_graph=True,
            write_images=False, 
            write_steps_per_second=False, 
            update_freq='epoch',
            profile_batch=0, 
            embeddings_freq=0, 
            embeddings_metadata=None
        )

    def draw_model(self, target):
        if self.model:
            plot_model(self.model, target)
        else:
            raise Exception("model needs to be set, use BaseModel.set_model(<model>)")

    def train_model(self, generator_train, generator_validation, steps_train=16, steps_validation=16, epochs=100):
        if self.model and self.checkpoint and self.tensorboard:
            callbacks = [self.tensorboard, self.checkpoint]

            return self.model.fit(
                generator_train,
                validation_data=generator_validation,
                epochs=epochs,
                steps_per_epoch=steps_train,
                validation_steps=steps_validation,
                callbacks=callbacks
            )

        else:
            raise Exception(
                """
                model needs to be set, use BaseModel.set_model(<model>), model: {}
                checkpoint needs to be set, use BaseModel.set_callbacks(<ModelCheckpoint>), checkpoint: {}
                tensorboard needs to be set, use BaseModel.set_callbacks(<TensorBoard>), tensorboard: {}
                """.format(self.model, self.checkpoint, self.tensorboard)
            )
            
    def create_train_generator(self, directory: str, batch_size=32, target_size=(640, 360), rotation_range=15,
                               width_shift_range=0.05, height_shift_range=0.05, shear_range=0.05, zoom_range=0.1,
                               brightness_range=(0.5, 1), channel_shift_range=0.1):

        image_generator_settings = ImageDataGenerator(
            rescale=1. / 255.,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=False,
            brightness_range=brightness_range,
            channel_shift_range=channel_shift_range,
            fill_mode='nearest')

        image_generator = image_generator_settings.flow_from_directory(
            directory=directory,
            target_size=target_size,
            color_mode="rgb",
            classes=None,
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=True,
            interpolation="nearest"
        )

        return image_generator

    def create_test_generator(self, directory: str, batch_size=32, target_size=(640, 360)):
        image_generator_settings = ImageDataGenerator(
            rescale=1. / 255.,
            fill_mode='nearest',
        )

        image_generator = image_generator_settings.flow_from_directory(
            directory=directory,
            target_size=target_size,
            color_mode="rgb",
            classes=None,
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=True,
            interpolation="nearest"
        )

        return image_generator


    def get_model(self, h5_file: str=None) -> Model:
        if h5_file:
            return load_model(h5_file)
        
        image_size = (640, 360, 3)
        label_count = len([200,"bad"])
        drop_rate = 0.2

        def block_res(image, filters, size=(3,3)):
            conv2d1_2 = layers.Conv2D(filters=filters,
                            kernel_size=size,
                            activation='relu',
                            padding='same')(image)
            conv2d2_2 = layers.Conv2D(filters=filters,
                            kernel_size=size,
                            activation='relu',
                            padding='same')(conv2d1_2)
            upsample1 = layers.Conv2D(filters=filters,
                            kernel_size=(1,1),
                            activation='relu',
                            padding='same')(image)
            add1 = layers.add([conv2d2_2, upsample1])
            max_pool1 = layers.MaxPooling2D()(add1)
            norm1 = layers.BatchNormalization()(max_pool1)
            return norm1
        

        input = layers.Input(image_size, name='image')
        block = block_res(input, 16)
        drop = layers.Dropout(drop_rate)(block)

        for i in range(0, 7):
            block = block_res(drop, 32 * (i + 1))
            drop = layers.Dropout(drop_rate)(block)

        conv2d1_1_1 = layers.Conv2D(filters=512,
            kernel_size=(2, 1),
            activation='relu',
            padding='valid')(drop)
        flatten1_1 = layers.Flatten()(conv2d1_1_1)
        dense1 = layers.Dense(units=512, activation='relu')(flatten1_1)
        output = layers.Dense(units=label_count, activation='softmax')(dense1)

        model = Model(inputs=[input], outputs=[output]) 

        optimizer = optimizers.RMSprop(lr=0.001)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy', 'mse'])

        return model




if __name__ == "__main__":
    train_path = "./data/"
    logpath = "./log"
    modelpath = "./model/test_640_360_3-2.h5"
    vg = vgg(train_path, logpath, modelpath, modelpath)
    vg.model.summary()

    train = vg.create_train_generator(train_path)
    test = vg.create_test_generator(train_path)
    vg.train_model(train, test)
    # https://www.section.io/engineering-education/image-classifier-keras/