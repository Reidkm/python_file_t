# TOF 
## 障碍物与空间检测
### 初始化
'''
在 tof_utils.pyx 所在文件夹
运行
python setup.py build_ext --inplace

进入文件夹 tof_obstacle_detection
运行命令
python setup.py build_ext --inplace
'''
### 使用
'''
使用图片
python tof_obstacle_detection_image_test.py

使用tof摄像头
python tof_obstacle_detection_live_test.py
'''