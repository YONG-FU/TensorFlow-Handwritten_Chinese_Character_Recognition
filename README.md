# Handwritten Chinese Character Recognition

Environment: Python + TensorFlow + OpenCV3

���˵Ļ�����Windows����������������Win7 64λ�ģ����Ե�JetBrains�Ĺٷ���վ����PyCharm IDE�����ߵİ汾��Ϣ���£�

python-3.6.3
pycharm-2017.2.4

���һ��˳�������Python���Կ���������IDE��׼��������������ǰ�װOpenCV3.3�Ŀ���������Ȼ��֧��Python���Ե�SDK��OpenCV�ٷ����ص�3.3�Ŀ��������������SDK�ǻ���Python2���޷����������������ʹ�ã���Ȼ�������Լ�ͨ��CMake���룬��������Ȼ���ǳ�ѧ�ߵĺ�ѡ��Python3֧��pip��ʽ�Զ���װ������������������ֻҪ��windows����������й��ߣ������������

pip install opencv-python ��װ���µ�OpenCV3.4������
pip install opencv-contrib-python ��װ���µ�OpenCV3.3��չ

���ݼ��������п�Ժ�Զ����о�������������:

http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip
http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip

��ѹ������һЩgnt�ļ���Ȼ�����˶������è����Ĵ��룬�������ļ���ת��Ϊ��ӦlabelĿ¼�µ�����png��ͼƬ����ע����HWDB1.1trn_gnt.zip��ѹ����alz�ļ�����Ҫ�ٴν�ѹ ��windows����alz�Ľ�ѹ����)��

https://pan.baidu.com/s/1o84jIrg#list/path=%2F%E5%AD%A6%E4%B9%A0%2F%E8%B5%84%E6%BA%90%E6%96%87%E6%A1%A3%2F%E6%95%B0%E6%8D%AE%E9%9B%86%2FHWDB1&parentPath=%2F%E5%AD%A6%E4%B9%A0%2F%E8%B5%84%E6%BA%90%E6%96%87%E6%A1%A3%2F%E6%95%B0%E6%8D%AE%E9%9B%86

A simple CNN with 4 convolutional layers and 2 fully-connected layers with dropout. Accuracy on test set is 93%. In my practice, I trained 100k steps on GTX1080, it take 10 hours and turns out that the accuracy is still gradually increasing, so I guess that you can acquire better accuracy by adding couple layers and training more steps. You are more than welcome to train from my checkpoint and help me to increase the accuracy even 0.1%, I would greatly appreciate it.


References:
https://zhuanlan.zhihu.com/p/24698483?refer=burness-DL
http://blog.topspeedsnail.com/archives/10897

