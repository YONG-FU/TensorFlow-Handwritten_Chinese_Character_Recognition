from __future__ import print_function
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Model, load_model

data_dir = r'../Datasets'
train_data_dir = os.path.join(data_dir, 'train')
test_data_dir = os.path.join(data_dir, 'test')

# dimensions of our images.
image_classes = 3755
image_width, image_height = 64, 64


def train(model):
    train_data_generator = ImageDataGenerator(rescale=1./255)
    test_data_generator = ImageDataGenerator(rescale=1./255)

    train_generator = train_data_generator.flow_from_directory(
        directory=train_data_dir,
        target_size=(image_width, image_height),
        batch_size=3000,
        color_mode='grayscale',
        class_mode='categorical')

    validation_generator = test_data_generator.flow_from_directory(
        directory=test_data_dir,
        target_size=(image_width, image_height),
        batch_size=3000,
        color_mode='grayscale',
        class_mode='categorical')

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=300,
        epochs=50,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=80)


def build_model(include_top=True, input_shape=(64, 64, 1), classes=image_classes):
    img_input = Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool')(x)

    if include_top:
        x = Flatten(name='flatten')(x)
        x = Dropout(0.05)(x)
        x = Dense(1024, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)

        model = Model(img_input, x, name='model')
        return model


model = build_model()
# model = load_model("./model.h5")
train(model)
model.save("./model.h5")