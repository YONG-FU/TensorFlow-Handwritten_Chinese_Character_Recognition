# -*- coding:utf-8 -*-
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing import image

# 读取预测模型
model = load_model("/aiml/code/model.h5")

# 图片加载预处理
image_directory = "/aiml/data/"
image_width, image_height = 64, 64
result_character = ""

for image_index in range(1, 501):
    image_path = image_directory + str(image_index) + ".png"

    x = image.load_img(image_path, grayscale=True, target_size=(image_width, image_height))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)

    # 进行图片预测
    prediction = model.predict(x)

    # 读取汉字标签
    with open("/aiml/code/word_dict", "rb") as word_dict_file:
        word_dict = pickle.load(word_dict_file)
        # print(word_dict["18"])
        # '18': '东'

    for label_index, label_value in enumerate(prediction[0]):
        if label_value == 1:
            print(str(image_index) + "." + word_dict[str(label_index)] + " ")
            result_character += word_dict[str(label_index)]

            # 写入预测内容
            with open("/aiml/result/result.txt", "w", encoding="utf-8") as result_file:
                result_file.write(result_character)