# ============================================================
# 1. Простая нейронная сеть, которая способна сложить 2 + 2
# ============================================================

# Данная сеть способна сложить 2+2 или другие небольшие значения

w0 = 0.9907079
w1 = 1.0264927
w2 = 0.01417504
w3 = -0.8950311
w4 = 0.88046944
w5 = 0.7524377
w6 = 0.794296
w7 = 1.1687347
w8 = 0.2406084

b0 = -0.00070612
b1 = -0.06846002
b2 = -0.00055442
b3 = -0.00000929


def relu(x):
    return max(0, x)


def predict(x1, x2):
    h1 = (x1 * w0) + (x2 * w1) + b0
    h2 = (x1 * w2) + (x2 * w3) + b1
    h3 = (x1 * w4) + (x2 * w5) + b2

    y = (relu(h1) * w6) + (relu(h2) * w7) + (relu(h3) * w8) + b3
    return y


print(predict(2, 2))
print(predict(1.5, 1.5))


# ============================================================
# 2. Загрузка изображения, преобразование в массив, нормализация
# ============================================================

# 1. Загрузка изображения
# 2. Масштабирование
# 3. Нормализация
# 4. Выбор модели
# 5. Загрузка изображения в модель и получение предсказания

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

img_path = "./digital_python-25-26/data/cat.png"

img = image.load_img(img_path, target_size=(224, 224))

# plt.imshow(img)
# plt.show()

img_array = image.img_to_array(img)
print(img_array.shape)

print(img_array[100, 100])

print(np.min(img_array))
print(np.max(img_array))

img_batch = np.expand_dims(img_array, axis=0)

from tensorflow.keras.applications.resnet50 import preprocess_input

img_preprocessed = preprocess_input(img_batch)
print(img_preprocessed.shape)

print(img_preprocessed[0, 100, 100])

print(np.min(img_preprocessed))
print(np.max(img_preprocessed))


# ============================================================
# 3. Обучение своей модели на основе MobileNet
# ============================================================

# Название папок = название категории

TRAIN_DATA_DIR = "./digital_python-25-26/data/train_data"
VALIDATION_DATA_DIR = "./digital_python-25-26/data/val_data"
TRAIN_SAMPLES = 500
VALIDATION_SAMPLES = 500

# "кошка или собака" -> "кошка или НЕ кошка" - бинарная классификация
# "кошка или собака" - мультиклассовая классификация
NUM_CLASSES = 2

IMG_WIDTH = 224
IMG_HEIGHT = 224

# Сколько изображений модель при обучении принимает одновременно
BATCH_SIZE = 64

# Аугментация - процедура увеличения кол-ва данных путем их "искажение":
# повороты, сдвиги, масштабирования

from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import (
    Input,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import math

# аугментация и нормализация
train_datagen = image.ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,  # 500 * 20 * 2 = 20 000
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
)

# только нормализация
val_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode="categorical",
)

val_gen = train_datagen.flow_from_directory(
    VALIDATION_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode="categorical",
)

model = MobileNet()
for layer in model.layers[:]:
    layer.trainable = False

input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))

custom_model = model(input)
custom_model = GlobalAveragePooling2D()(custom_model)
custom_model = Dense(64, activation="relu")(custom_model)
custom_model = Dropout(0.5)(custom_model)
prediction = Dense(NUM_CLASSES, activation="softmax")(custom_model)

target_model = Model(inputs=input, outputs=prediction)

target_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(),
    metrics=["acc"],
)

num_steps = math.ceil(float(TRAIN_SAMPLES) / BATCH_SIZE)
target_model.fit(
    train_gen,
    steps_per_epoch=num_steps,
    epochs=7,
    validation_data=val_gen,
    validation_steps=num_steps,
)

print(val_gen.class_indices)

target_model.save("./digital_python-25-26/data/our_model.h5")


# ============================================================
# 4. Загрузка сохраненной модели и получение предсказания
# ============================================================

from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
import numpy as np

# img_path = "./digital_python-25-26/data/cat.png"
# img_path = "./digital_python-25-26/data/dog.png"
img_path = "./digital_python-25-26/data/luna.jpg"

img = image.load_img(img_path, target_size=(224, 224))

img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)

from tensorflow.keras.applications.resnet50 import preprocess_input

img_preprocessed = preprocess_input(img_batch)

img = image.load_img(img_path, target_size=(224, 224))

model = load_model("./digital_python-25-26/data/our_model.h5")

prediction = model.predict(img_preprocessed)

print(prediction)
