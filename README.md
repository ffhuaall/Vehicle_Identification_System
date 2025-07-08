🚗 车辆识别系统

基于YOLOv8的车辆检测系统，支持图片和视频的实时车辆识别。

项目介绍

本项目是一个智能车辆检测系统，具有以下特点：
- 基于YOLOv8深度学习模型进行车辆检测
- 支持图片和视频文件的车辆识别
- 提供PyQt5图形用户界面
- 自动保存检测结果

🎯 快速启动

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 启动程序

运行GUI界面：
```bash
python vehicle_detection_gui.py
```

训练模型（可选）：
```bash
python vehicle_detection_yolov8.py
```

3. 使用方法

1. 启动GUI后，点击"选择目录"按钮
2. 选择包含图片或视频的文件夹
3. 系统会自动进行车辆检测并显示结果
4. 检测结果会保存在`detection_results/`目录中

项目结构

```
├── vehicle_detection_gui.py        # GUI主程序
├── vehicle_detection_yolov8.py     # 模型训练脚本
├── best.pt                         # 训练好的模型
├── requirements.txt                # 依赖包列表
├── imagetest/                      # 测试图片
├── mediatest/                      # 测试视频
└── detection_results/              # 检测结果
```

环境要求

- Python 3.7+
- 主要依赖：ultralytics, PyQt5, opencv-python, torch
