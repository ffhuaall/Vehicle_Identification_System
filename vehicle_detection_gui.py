#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆识别GUI界面
"""

import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                            QFileDialog, QHBoxLayout, QVBoxLayout, QSizePolicy, QSlider)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from ultralytics import YOLO


def vehicle_detect_image(path):
    """
    使用YOLO模型对图像进行车辆检测

    参数:
        path (str): 要检测的图片路径

    返回:
        tuple: (检测结果描述, 检测后图片保存路径)
    """
    try:
        model = YOLO("./best.pt")  # 加载预训练的车辆检测模型
        results = model(path)  # 对图像进行预测

        # 创建检测结果保存目录
        output_dir = "detection_results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 生成输出文件名
        filename = os.path.basename(path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"detected_{name}{ext}")

        # 处理预测结果
        for result in results:
            boxes = result.boxes  # 获取边界框信息
            
            # 保存带检测框的图片
            result_img = result.plot()  # 绘制检测结果
            cv2.imwrite(output_path, result_img)
            
            if boxes is not None and len(boxes) > 0:
                # 统计检测到的车辆数量
                vehicle_count = len(boxes)
                
                # 获取所有检测到的车辆类别
                detected_vehicles = []
                for box in boxes:
                    cls_id = int(box.cls[0].item())  # 类别ID转换为整数
                    class_name = result.names[cls_id]  # 获取类别名称
                    confidence = float(box.conf[0].item())  # 获取置信度
                    detected_vehicles.append(f"{class_name}({confidence:.2f})")
                
                # 返回检测结果
                if vehicle_count == 1:
                    return f"检测到 1 辆车辆: {detected_vehicles[0]}", output_path
                else:
                    return f"检测到 {vehicle_count} 辆车辆: {', '.join(detected_vehicles[:3])}{'...' if len(detected_vehicles) > 3 else ''}", output_path
            else:
                return "未检测到车辆", output_path
                
    except Exception as e:
        return f"检测出错: {str(e)}", None


def vehicle_detect_frame(frame, model):
    """
    使用YOLO模型对视频帧进行车辆检测

    参数:
        frame: 视频帧
        model: YOLO模型

    返回:
        tuple: (检测结果描述, 处理后的帧)
    """
    try:
        results = model(frame)
        
        for result in results:
            boxes = result.boxes
            
            # 绘制检测结果
            result_frame = result.plot()
            
            if boxes is not None and len(boxes) > 0:
                vehicle_count = len(boxes)
                detected_vehicles = []
                
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    class_name = result.names[cls_id]
                    confidence = float(box.conf[0].item())
                    detected_vehicles.append(f"{class_name}({confidence:.2f})")
                
                if vehicle_count == 1:
                    return f"检测到 1 辆车辆: {detected_vehicles[0]}", result_frame
                else:
                    return f"检测到 {vehicle_count} 辆车辆: {', '.join(detected_vehicles[:3])}{'...' if len(detected_vehicles) > 3 else ''}", result_frame
            else:
                return "未检测到车辆", result_frame
                
    except Exception as e:
        return f"检测出错: {str(e)}", frame


class VideoThread(QThread):
    """
    视频播放线程
    """
    frame_ready = pyqtSignal(np.ndarray, str)  # 发送帧和检测结果
    
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.cap = None
        self.model = None
        self.is_playing = False
        self.is_paused = False
        
    def run(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.model = YOLO("./best.pt")
        
        if not self.cap.isOpened():
            return
            
        self.is_playing = True
        
        while self.is_playing:
            if not self.is_paused:
                ret, frame = self.cap.read()
                if ret:
                    # 进行车辆检测
                    detection_result, processed_frame = vehicle_detect_frame(frame, self.model)
                    self.frame_ready.emit(processed_frame, detection_result)
                    self.msleep(33)  # 约30fps
                else:
                    # 视频结束，重新开始
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                self.msleep(0.1)
                
    def pause(self):
        self.is_paused = True
        
    def resume(self):
        self.is_paused = False
        
    def stop(self):
        self.is_playing = False
        if self.cap:
            self.cap.release()
        self.quit()
        self.wait()


class VehicleDetectionViewer(QWidget):
    def __init__(self):
        super().__init__()
        # 窗口设置
        self.setWindowTitle("车辆识别系统 - 支持图片和视频")
        self.setGeometry(300, 300, 1200, 900)

        # 文件列表
        self.file_list = []
        self.current_index = 0
        self.current_file_type = None  # 'image' 或 'video'
        
        # 视频相关
        self.video_thread = None
        self.is_video_playing = False

        # 创建UI
        self.create_ui()

        # 设置定时器（用于图片轮播）
        self.timer = QTimer(self)
        self.timer.setInterval(3000)  # 3秒间隔
        self.timer.timeout.connect(self.show_next_file)

    def create_ui(self):
        # 主布局 - 水平布局
        main_layout = QHBoxLayout(self)

        # 左侧图片区域 (70%宽度)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.image_label = QLabel("车辆检测图片显示区域")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            background-color: #f5f5f5;
            border: 2px solid #2196F3;
            border-radius: 10px;
            font-size: 18px;
            color: #666;
            font-weight: bold;
        """)
        self.image_label.setMinimumHeight(450)
        left_layout.addWidget(self.image_label)

        main_layout.addWidget(left_panel, 7)  # 7份宽度

        # 右侧控制面板 (30%宽度)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setAlignment(Qt.AlignTop)

        # 标题标签
        title_label = QLabel("🚗 车辆识别系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #2196F3;
            margin-bottom: 20px;
            padding: 10px;
        """)
        right_layout.addWidget(title_label)

        # 选择目录按钮
        self.dir_button = QPushButton("📁 选择媒体目录")
        self.dir_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        self.dir_button.clicked.connect(self.select_directory)
        right_layout.addWidget(self.dir_button)

        # 检测结果标签
        self.info_label = QLabel("检测结果将显示在这里")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("""
            font-size: 18px;
            color: #333;
            background-color: #e8f5e8;
            border: 2px solid #4CAF50;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            min-height: 120px;
        """)
        self.info_label.setWordWrap(True)
        right_layout.addWidget(self.info_label)

        # 控制按钮区域
        control_layout = QVBoxLayout()
        
        # 暂停/继续按钮
        self.pause_button = QPushButton("⏸️ 暂停播放")
        self.pause_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        self.pause_button.clicked.connect(self.toggle_playback)
        self.pause_button.setEnabled(False)
        control_layout.addWidget(self.pause_button)

        # 手动切换按钮
        self.next_button = QPushButton("⏭️ 下一个")
        self.next_button.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        self.next_button.clicked.connect(self.show_next_file)
        self.next_button.setEnabled(False)
        control_layout.addWidget(self.next_button)

        right_layout.addLayout(control_layout)

        # 状态信息标签
        self.status_label = QLabel("状态: 等待选择目录")
        self.status_label.setStyleSheet("""
            font-size: 16px;
            color: #666;
            background-color: #f0f0f0;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 12px;
            margin-top: 10px;
        """)
        right_layout.addWidget(self.status_label)

        # 添加空白区域以填充空间
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(spacer)

        main_layout.addWidget(right_panel, 3)  # 3份宽度

    def select_directory(self):
        """选择媒体文件目录"""
        # 打开目录选择对话框
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择包含图片或视频的目录",
            os.path.expanduser("~"),  # 默认从用户主目录开始
            QFileDialog.ShowDirsOnly
        )

        if directory:
            # 加载目录中的媒体文件
            self.load_media_files(directory)

    def load_media_files(self, directory):
        """加载指定目录中的图片和视频文件"""
        # 获取目录中所有媒体文件
        self.file_list = []
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if filename.lower().endswith(image_extensions):
                self.file_list.append((file_path, 'image'))
            elif filename.lower().endswith(video_extensions):
                self.file_list.append((file_path, 'video'))

        if self.file_list:
            # 重置索引并显示第一个文件
            self.current_index = 0
            self.show_current_file()
            
            # 更新状态和启用按钮
            image_count = sum(1 for _, ftype in self.file_list if ftype == 'image')
            video_count = sum(1 for _, ftype in self.file_list if ftype == 'video')
            self.status_label.setText(f"状态: 已加载 {image_count} 张图片, {video_count} 个视频")
            self.pause_button.setEnabled(True)
            self.next_button.setEnabled(True)

            # 如果第一个文件是图片，启动轮播
            if self.file_list[0][1] == 'image':
                self.timer.start()
        else:
            self.image_label.setText("❌ 没有找到媒体文件")
            self.info_label.setText("请选择包含图片或视频的目录")
            self.status_label.setText("状态: 目录中无媒体文件")

    def show_current_file(self):
        """显示当前文件（图片或视频）并进行车辆检测"""
        if self.file_list:
            file_path, file_type = self.file_list[self.current_index]
            filename = os.path.basename(file_path)
            
            # 停止之前的视频播放
            self.stop_video()
            
            self.current_file_type = file_type
            
            if file_type == 'image':
                self.show_image(file_path, filename)
            elif file_type == 'video':
                self.show_video(file_path, filename)
                
    def show_image(self, image_path, filename):
        """显示图片并进行车辆检测"""
        # 更新状态
        self.status_label.setText(f"状态: 正在检测 {self.current_index + 1}/{len(self.file_list)} - {filename}")

        try:
            # 进行车辆检测
            self.info_label.setText("🔍 正在检测车辆...")
            QApplication.processEvents()  # 更新界面
            
            detection_result, detected_image_path = vehicle_detect_image(image_path)
            
            # 显示检测后的图片（带框和置信度）
            if detected_image_path and os.path.exists(detected_image_path):
                pixmap = QPixmap(detected_image_path)
            else:
                # 如果检测图片不存在，显示原图
                pixmap = QPixmap(image_path)

            if not pixmap.isNull():
                # 缩放图片以适应标签大小
                scaled_pixmap = pixmap.scaled(
                    self.image_label.width() - 20,
                    self.image_label.height() - 20,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
                
                # 显示检测结果
                result_text = f"📷 图片: {filename}\n\n🚗 检测结果:\n{detection_result}"
                self.info_label.setText(result_text)

            else:
                self.image_label.setText("❌ 无法加载图片")
                self.info_label.setText(f"无效图片: {filename}")
                
        except Exception as e:
            self.image_label.setText("❌ 加载图片出错")
            self.info_label.setText(f"错误: {str(e)}")
            
    def show_video(self, video_path, filename):
        """显示视频并进行车辆检测"""
        # 更新状态
        self.status_label.setText(f"状态: 正在播放 {self.current_index + 1}/{len(self.file_list)} - {filename}")
        
        try:
            # 创建视频播放线程
            self.video_thread = VideoThread(video_path)
            self.video_thread.frame_ready.connect(self.update_video_frame)
            self.video_thread.start()
            
            self.is_video_playing = True
            
            # 显示视频信息
            self.info_label.setText(f"🎬 视频: {filename}\n\n🚗 检测结果:\n正在加载...")
            
        except Exception as e:
            self.image_label.setText("❌ 无法播放视频")
            self.info_label.setText(f"错误: {str(e)}")
            
    def update_video_frame(self, frame, detection_result):
        """更新视频帧显示"""
        try:
            # 将OpenCV帧转换为QPixmap
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)
            
            # 缩放以适应标签大小
            scaled_pixmap = pixmap.scaled(
                self.image_label.width() - 20,
                self.image_label.height() - 20,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            
            # 更新检测结果
            if self.file_list:
                filename = os.path.basename(self.file_list[self.current_index][0])
                result_text = f"🎬 视频: {filename}\n\n🚗 检测结果:\n{detection_result}"
                self.info_label.setText(result_text)
                
        except Exception as e:
            print(f"更新视频帧错误: {e}")
            
    def stop_video(self):
        """停止视频播放"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread = None
        self.is_video_playing = False

    def show_next_file(self):
        """显示下一个文件"""
        if self.file_list:
            self.current_index = (self.current_index + 1) % len(self.file_list)
            self.show_current_file()
            
            # 如果切换到图片，启动定时器；如果是视频，停止定时器
            if self.file_list[self.current_index][1] == 'image':
                if not self.timer.isActive():
                    self.timer.start()
            else:
                if self.timer.isActive():
                    self.timer.stop()

    def toggle_playback(self):
        """切换播放状态"""
        if self.current_file_type == 'image':
            # 图片轮播控制
            if self.timer.isActive():
                self.timer.stop()
                self.pause_button.setText("▶️ 继续播放")
                self.pause_button.setStyleSheet("""
                    QPushButton {
                        background-color: #4CAF50;
                        color: white;
                        border: none;
                        border-radius: 6px;
                        padding: 12px;
                        font-size: 16px;
                    }
                    QPushButton:hover {
                        background-color: #45a049;
                    }
                """)
            else:
                self.timer.start()
                self.pause_button.setText("⏸️ 暂停播放")
                self.pause_button.setStyleSheet("""
                    QPushButton {
                        background-color: #FF9800;
                        color: white;
                        border: none;
                        border-radius: 6px;
                        padding: 12px;
                        font-size: 16px;
                    }
                    QPushButton:hover {
                        background-color: #F57C00;
                    }
                """)
        elif self.current_file_type == 'video':
            # 视频播放控制
            if self.video_thread and self.video_thread.is_paused:
                self.video_thread.resume()
                self.pause_button.setText("⏸️ 暂停播放")
                self.pause_button.setStyleSheet("""
                    QPushButton {
                        background-color: #FF9800;
                        color: white;
                        border: none;
                        border-radius: 6px;
                        padding: 12px;
                        font-size: 16px;
                    }
                    QPushButton:hover {
                        background-color: #F57C00;
                    }
                """)
            elif self.video_thread:
                self.video_thread.pause()
                self.pause_button.setText("▶️ 继续播放")
                self.pause_button.setStyleSheet("""
                    QPushButton {
                        background-color: #4CAF50;
                        color: white;
                        border: none;
                        border-radius: 6px;
                        padding: 12px;
                        font-size: 16px;
                    }
                    QPushButton:hover {
                        background-color: #45a049;
                    }
                """)

    def resizeEvent(self, event):
        """窗口大小改变时重新调整显示内容大小"""
        super().resizeEvent(event)
        if self.file_list:
            # 延迟重新显示当前文件，避免频繁更新
            QTimer.singleShot(100, self.show_current_file)
            
    def closeEvent(self, event):
        """窗口关闭时清理资源"""
        self.stop_video()
        if self.timer.isActive():
            self.timer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用程序图标和样式
    app.setStyle('Fusion')
    
    window = VehicleDetectionViewer()
    window.show()
    
    sys.exit(app.exec_())