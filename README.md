# Handwritten Chinese Character Recognition

Environment: Python + TensorFlow + OpenCV3

本人的环境是Windows开发环境，机器是Win7 64位的，可以到JetBrains的官方网站下载PyCharm IDE。工具的版本信息如下：

python-3.6.3
pycharm-2017.2.4

如果一切顺利，你的Python语言开发环境与IDE就准备好啦，下面就是安装OpenCV3.3的开发包，当然是支持Python语言的SDK，OpenCV官方下载的3.3的开发包里面包含的SDK是基于Python2，无法在我们这种情况下使用，当然还可以自己通过CMake编译，但是这显然不是初学者的好选择，Python3支持pip方式自动安装第三方开发包，我们只要打开windows下面的命令行工具，输入如下命令：

pip install opencv-python 安装最新的OpenCV3.4开发包
pip install opencv-contrib-python 安装最新的OpenCV3.3扩展

数据集来自于中科院自动化研究所，具体下载:

http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip
http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip

解压后发现是一些gnt文件，然后用了斗大的熊猫里面的代码，将所有文件都转化为对应label目录下的所有png的图片。（注意在HWDB1.1trn_gnt.zip解压后是alz文件，需要再次解压 ，windows上有alz的解压工具)。

https://pan.baidu.com/s/1o84jIrg#list/path=%2F%E5%AD%A6%E4%B9%A0%2F%E8%B5%84%E6%BA%90%E6%96%87%E6%A1%A3%2F%E6%95%B0%E6%8D%AE%E9%9B%86%2FHWDB1&parentPath=%2F%E5%AD%A6%E4%B9%A0%2F%E8%B5%84%E6%BA%90%E6%96%87%E6%A1%A3%2F%E6%95%B0%E6%8D%AE%E9%9B%86

A simple CNN with 4 convolutional layers and 2 fully-connected layers with dropout. Accuracy on test set is 93%. In my practice, I trained 100k steps on GTX1080, it take 10 hours and turns out that the accuracy is still gradually increasing, so I guess that you can acquire better accuracy by adding couple layers and training more steps. You are more than welcome to train from my checkpoint and help me to increase the accuracy even 0.1%, I would greatly appreciate it.


References:
https://zhuanlan.zhihu.com/p/24698483?refer=burness-DL
http://blog.topspeedsnail.com/archives/10897

