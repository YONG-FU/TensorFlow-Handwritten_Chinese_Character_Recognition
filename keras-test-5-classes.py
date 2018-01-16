# -*- coding:utf-8 -*-
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

image_path = r"222498.png"
image_classes = 5
image_width, image_height = 64, 64

image = image.load_img(image_path, grayscale=True, target_size=(image_width, image_height))
x = image.img_to_array(image)
x = np.expand_dims(x, axis=0)

model = load_model("model.h5")
prediction = model.predict(x)

# 输出预测类别
print(prediction)
