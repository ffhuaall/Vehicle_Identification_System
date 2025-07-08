#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8车辆识别训练脚本
基于自定义数据集训练YOLOv8模型进行车辆检测
"""

import os
import xml.etree.ElementTree as ET
import yaml
from pathlib import Path
import shutil
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class VehicleDetectionTrainer: # 核心训练类
    def __init__(self, data_root="testdata/MyData", output_dir="yolo_dataset"):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.classes = ['car']  # 根据XML文件中的类别定义
        
        # 创建YOLO格式的目录结构
        self.yolo_dirs = {
            'train': self.output_dir / 'train',
            'val': self.output_dir / 'val', 
            'test': self.output_dir / 'test'
        }
        
        for split in self.yolo_dirs.values():
            (split / 'images').mkdir(parents=True, exist_ok=True)
            (split / 'labels').mkdir(parents=True, exist_ok=True)
    
    def convert_bbox_to_yolo(self, bbox, img_width, img_height):
        """
        将Pascal VOC格式的边界框转换为YOLO格式
        VOC格式: (xmin, ymin, xmax, ymax)
        YOLO格式: (x_center, y_center, width, height) 归一化到[0,1]
        """
        xmin, ymin, xmax, ymax = bbox
        
        # 计算中心点和宽高
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin
        
        # 归一化
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        return x_center, y_center, width, height
    
    def parse_xml_annotation(self, xml_path):
        """
        解析XML标注文件
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 获取图像尺寸
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        # 获取所有目标
        objects = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name in self.classes:
                class_id = self.classes.index(class_name)
                
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                # 转换为YOLO格式
                yolo_bbox = self.convert_bbox_to_yolo(
                    (xmin, ymin, xmax, ymax), img_width, img_height
                )
                
                objects.append((class_id, *yolo_bbox))
        
        return objects, img_width, img_height
    
    def convert_dataset(self):
        """
        将Pascal VOC格式数据集转换为YOLO格式
        """
        print("开始转换数据集格式...")
        
        # 处理训练集、验证集和测试集
        splits = ['train', 'val', 'test']
        
        for split in splits:
            split_dir = self.data_root / split
            if not split_dir.exists():
                print(f"警告: {split_dir} 目录不存在，跳过")
                continue
                
            images_dir = split_dir / 'images'
            annotations_dir = split_dir / 'annotations'
            
            if not images_dir.exists() or not annotations_dir.exists():
                print(f"警告: {split} 目录结构不完整，跳过")
                continue
            
            print(f"处理 {split} 数据集...")
            
            # 获取所有图像文件
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            
            for img_path in image_files:
                # 对应的XML文件
                xml_path = annotations_dir / (img_path.stem + '.xml')
                
                if not xml_path.exists():
                    print(f"警告: 找不到对应的标注文件 {xml_path}")
                    continue
                
                try:
                    # 解析XML标注
                    objects, img_width, img_height = self.parse_xml_annotation(xml_path)
                    
                    if not objects:
                        print(f"警告: {xml_path} 中没有有效的目标")
                        continue
                    
                    # 复制图像文件
                    dst_img_path = self.yolo_dirs[split] / 'images' / img_path.name
                    shutil.copy2(img_path, dst_img_path)
                    
                    # 创建YOLO格式的标注文件
                    label_path = self.yolo_dirs[split] / 'labels' / (img_path.stem + '.txt')
                    with open(label_path, 'w') as f:
                        for obj in objects:
                            class_id, x_center, y_center, width, height = obj
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
                except Exception as e:
                    print(f"处理 {img_path} 时出错: {e}")
                    continue
            
            print(f"{split} 数据集处理完成")
        
        print("数据集转换完成！")
    
    def create_yaml_config(self):
        """
        创建YOLO训练配置文件
        """
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.classes),
            'names': self.classes
        }
        
        config_path = self.output_dir / 'dataset.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"配置文件已保存到: {config_path}")
        return config_path
    
    def train_model(self, epochs=100, img_size=640, batch_size=16, device=None, debug=False):
        """
        训练YOLOv8模型
        
        参数:
            epochs: 训练轮数
            img_size: 图像尺寸
            batch_size: 批次大小
            device: 训练设备，None表示自动选择
            debug: 是否启用调试模式
        """
        print("开始训练YOLOv8模型...")
        
        # 自动检测并选择最佳设备
        if device is None:
            import torch
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        try:
            if 'cuda' in device:
                # 如果使用GPU，打印GPU信息
                import torch
                if not torch.cuda.is_available():
                    print("警告: 指定了CUDA设备，但PyTorch报告CUDA不可用")
                    print("将回退到CPU训练")
                    device = 'cpu'
                else:
                    # 获取GPU设备索引
                    device_idx = int(device.split(':')[1]) if ':' in device else 0
                    if device_idx >= torch.cuda.device_count():
                        print(f"警告: 指定的GPU索引 {device_idx} 超出可用范围 (0-{torch.cuda.device_count()-1})")
                        print("将使用默认GPU (cuda:0)")
                        device = 'cuda:0'
                    
                    # 打印GPU信息
                    gpu_name = torch.cuda.get_device_name(device_idx)
                    gpu_memory = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
                    print(f"使用GPU训练: {gpu_name}, 显存: {gpu_memory:.2f}GB")
                    
                    # 根据GPU显存自动调整批次大小
                    if batch_size > 8:
                        if gpu_memory < 6:
                            # 小显存GPU (如4GB)
                            adjusted_batch = min(batch_size, 4)
                            if adjusted_batch != batch_size:
                                print(f"显存较小，自动调整批次大小: {batch_size} -> {adjusted_batch}")
                                batch_size = adjusted_batch
                        elif gpu_memory < 8:
                            # 中等显存GPU (6-8GB)
                            adjusted_batch = min(batch_size, 8)
                            if adjusted_batch != batch_size:
                                print(f"显存适中，自动调整批次大小: {batch_size} -> {adjusted_batch}")
                                batch_size = adjusted_batch
                    
                    # 验证CUDA是否真正可用
                    if debug:
                        print("验证CUDA可用性...")
                        test_tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
                        print(f"测试张量创建成功，设备: {test_tensor.device}")
            else:
                print("使用CPU训练，速度可能较慢")
                print("提示: 如果您有NVIDIA GPU，请确保正确安装CUDA和兼容的PyTorch版本")
        except Exception as e:
            print(f"设置训练设备时出错: {e}")
            print("将回退到CPU训练")
            device = 'cpu'
        
        try:
            # 加载预训练模型
            model = YOLO('yolov8n.pt')  # 使用nano版本，速度快
            
            # 创建配置文件
            config_path = self.create_yaml_config()
            
            # 训练参数
            train_args = {
                'data': str(config_path),
                'epochs': epochs,
                'imgsz': img_size,
                'batch': batch_size,
                'name': 'vehicle_detection',
                'save': True,
                'plots': True,
                'device': device,
                'workers': 4 if 'cuda' in device else 0,  # GPU训练时使用多个工作进程
                'amp': True  # 使用混合精度训练加速
            }
            
            # 如果是调试模式，添加额外参数
            if debug:
                train_args['verbose'] = True
            
            # 开始训练
            print(f"\n使用设备 '{device}' 开始训练...")
            results = model.train(**train_args)
            
            print("训练完成！")
            return model, results
            
        except Exception as e:
            print(f"\n训练过程中出错: {e}")
            
            if 'cuda' in device:
                print("\n===== GPU训练失败，可能的原因 =====")
                print("1. GPU显存不足 - 尝试减小批次大小 (--batch_size)")
                print("2. CUDA版本与PyTorch不兼容 - 重新安装匹配的版本")
                print("3. GPU驱动程序过旧 - 更新显卡驱动")
                print("4. 其他CUDA相关错误")
                print("\n尝试使用CPU重新训练: python vehicle_detection_yolov8.py --no_gpu")
            
            raise e
    
    def validate_model(self, model, device=None):
        """
        验证模型性能
        
        参数:
            model: 训练好的模型
            device: 验证设备，None表示使用模型训练时的设备
        """
        print("验证模型性能...")
        
        # 在验证集上评估
        results = model.val(device=device)
        
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
        
        return results
    
    def test_inference(self, model, test_image_path=None, device=None):
        """
        测试模型推理
        
        参数:
            model: 训练好的模型
            test_image_path: 测试图片路径，None表示使用验证集中的一张图片
            device: 推理设备，None表示使用模型训练时的设备
        """
        if test_image_path is None:
            # 使用验证集中的一张图片进行测试
            val_images = list((self.yolo_dirs['val'] / 'images').glob('*.jpg'))
            if val_images:
                test_image_path = val_images[0]
            else:
                print("没有找到测试图片")
                return
        
        print(f"测试图片: {test_image_path}")
        
        # 进行推理
        results = model(test_image_path, device=device)
        
        # 显示结果
        for r in results:
            # 保存结果图片
            output_path = "detection_result.jpg"
            r.save(output_path)
            print(f"检测结果已保存到: {output_path}")
            
            # 打印检测到的目标信息
            if r.boxes is not None:
                for box in r.boxes:
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    print(f"检测到 {self.classes[cls]}, 置信度: {conf:.4f}")
            else:
                print("未检测到任何目标")
    
    def run_complete_pipeline(self, epochs=50, img_size=640, batch_size=8, device=None, debug=False):
        """
        运行完整的训练流程
        
        参数:
            epochs: 训练轮数
            img_size: 图像尺寸
            batch_size: 批次大小
            device: 训练设备，None表示自动选择
            debug: 是否启用调试模式
        """
        print("=" * 50)
        print("YOLOv8车辆检测训练流程开始")
        print("=" * 50)
        
        # 1. 转换数据集格式
        self.convert_dataset()
        
        # 2. 训练模型
        model, train_results = self.train_model(epochs=epochs, img_size=img_size, batch_size=batch_size, device=device, debug=debug)
        
        # 3. 验证模型 (使用相同的设备)
        val_results = self.validate_model(model, device=device)
        
        # 4. 测试推理 (使用相同的设备)
        self.test_inference(model, device=device)
        
        print("=" * 50)
        print("训练流程完成！")
        print("=" * 50)
        
        return model

def check_gpu_availability():
    """
    检查GPU可用性并返回详细信息
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if not cuda_available:
            print("\n===== GPU不可用，可能的原因 =====")
            print("1. 没有安装CUDA或CUDA版本与PyTorch不兼容")
            print("2. 没有NVIDIA GPU或驱动程序未正确安装")
            print("3. PyTorch没有编译CUDA支持")
            
            # 检查PyTorch安装
            print("\n===== PyTorch信息 =====")
            print(f"PyTorch版本: {torch.__version__}")
            print(f"是否编译了CUDA支持: {'是' if torch.cuda.is_available() else '否'}")
            
            # 尝试获取CUDA版本
            try:
                print(f"CUDA版本: {torch.version.cuda if hasattr(torch.version, 'cuda') else '未知'}")
            except:
                print("无法获取CUDA版本信息")
                
            print("\n===== 解决方案 =====")
            print("1. 确保安装了NVIDIA GPU驱动程序")
            print("2. 安装与PyTorch兼容的CUDA版本")
            print("3. 重新安装带有CUDA支持的PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            print("4. 如果没有GPU，请使用--no_gpu参数运行程序")
            
            return False, "CPU", {}
        else:
            # GPU可用，获取详细信息
            gpu_count = torch.cuda.device_count()
            gpu_info = {}
            
            print("\n===== 检测到可用GPU =====")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                gpu_info[i] = {"name": gpu_name, "memory": f"{gpu_memory:.2f}GB"}
                print(f"GPU {i}: {gpu_name}, 显存: {gpu_memory:.2f}GB")
            
            return True, "cuda:0", gpu_info
    except Exception as e:
        print(f"\n检查GPU时出错: {e}")
        print("将使用CPU进行训练")
        return False, "cpu", {}

def main():
    """
    主函数
    """
    import argparse
    import torch
    import os
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='YOLOv8车辆检测训练程序')
    parser.add_argument('--data_root', type=str, default="testdata/MyData", help='数据集根目录')
    parser.add_argument('--output_dir', type=str, default="yolo_vehicle_dataset", help='输出目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--img_size', type=int, default=640, help='图像尺寸')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--device', type=str, default=None, 
                        help='训练设备 (例如: cpu, cuda:0, cuda:1), 默认自动选择')
    parser.add_argument('--no_gpu', action='store_true', help='强制使用CPU训练')
    parser.add_argument('--debug', action='store_true', help='启用调试模式，显示更多信息')
    parser.add_argument('--force_cuda', action='store_true', help='强制尝试使用CUDA，即使检测失败')
    
    args = parser.parse_args()
    
    # 设置环境变量，帮助调试CUDA问题
    if args.debug:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'
        print("已启用调试模式")
    
    # 设备选择逻辑
    if args.no_gpu:
        device = 'cpu'
        print("已指定使用CPU训练")
    elif args.device is not None:
        device = args.device
        print(f"使用指定设备: {device}")
        
        # 验证指定的设备是否可用
        if 'cuda' in device:
            try:
                device_idx = int(device.split(':')[1]) if ':' in device else 0
                if not torch.cuda.is_available() or device_idx >= torch.cuda.device_count():
                    print(f"警告: 指定的设备 {device} 不可用，将尝试使用默认GPU或CPU")
                    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            except Exception as e:
                print(f"设备指定错误: {e}，将使用默认设备")
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    elif args.force_cuda:  # 用户强制使用CUDA
        device = 'cuda:0'
        print("强制使用CUDA设备")
    else:
        # 详细检查GPU可用性
        gpu_available, default_device, gpu_info = check_gpu_availability()
        device = default_device
        print(f"自动选择设备: {device}")
        if not gpu_available:
            print("如果您确定系统有NVIDIA GPU，请尝试使用 --force_cuda 参数")
    
    # 如果使用CUDA，显示一些额外的诊断信息
    if (('cuda' in device and args.debug) or args.force_cuda):
        try:
            import torch
            print("\n===== CUDA诊断信息 =====")
            print(f"PyTorch版本: {torch.__version__}")
            print(f"CUDA是否可用: {torch.cuda.is_available()}")
            print(f"CUDA设备数量: {torch.cuda.device_count()}")
            
            if torch.cuda.is_available():
                print(f"当前CUDA设备: {torch.cuda.current_device()}")
                print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
                print(f"CUDA版本: {torch.version.cuda}")
                print(f"cuDNN版本: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else '不可用'}")
                print(f"cuDNN是否启用: {torch.backends.cudnn.enabled}")
                
                # 测试CUDA张量创建
                print("\n测试CUDA张量创建...")
                test_tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
                print(f"测试张量设备: {test_tensor.device}")
                
                # 测试简单运算
                result = test_tensor * 2
                print(f"简单运算测试成功: {result}")
                print("CUDA测试成功!")
            else:
                print("\n===== CUDA环境检查 =====")
                import subprocess
                
                # 检查NVIDIA驱动
                try:
                    if os.name == 'nt':  # Windows
                        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                        if result.returncode == 0:
                            print("NVIDIA驱动已安装")
                            print('\n'.join(result.stdout.split('\n')[0:6]))
                        else:
                            print("未检测到NVIDIA驱动或GPU")
                    else:  # Linux/Mac
                        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                        if result.returncode == 0:
                            print("NVIDIA驱动已安装")
                            print('\n'.join(result.stdout.split('\n')[0:6]))
                        else:
                            print("未检测到NVIDIA驱动或GPU")
                except Exception as e:
                    print(f"检查NVIDIA驱动失败: {e}")
                
                # 检查CUDA环境变量
                cuda_path = os.environ.get('CUDA_PATH')
                if cuda_path:
                    print(f"CUDA_PATH环境变量: {cuda_path}")
                else:
                    print("未设置CUDA_PATH环境变量")
                
                # 检查PyTorch CUDA构建
                if torch.cuda.is_available() == False:
                    print("\nPyTorch未检测到CUDA，可能的原因:")
                    print("1. PyTorch安装时没有CUDA支持")
                    print("2. CUDA版本与PyTorch不兼容")
                    print("3. GPU驱动程序过旧")
                    print("\n建议:")
                    print("- 重新安装带CUDA支持的PyTorch: https://pytorch.org/get-started/locally/")
                    print("- 更新NVIDIA驱动")
        except Exception as e:
            print(f"CUDA诊断过程中出错: {e}")
            print("这可能表明CUDA环境配置有问题，将回退到CPU训练")
            if not args.force_cuda:
                device = 'cpu'
    
    # 创建训练器实例
    trainer = VehicleDetectionTrainer(
        data_root=args.data_root,
        output_dir=args.output_dir
    )
    
    # 运行完整训练流程
    model = trainer.run_complete_pipeline(
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        device=device,
        debug=args.debug
    )
    
    print("\n训练完成！模型已保存在 runs/detect/vehicle_detection/ 目录下")
    print("可以使用以下代码加载训练好的模型：")
    print("model = YOLO('runs/detect/vehicle_detection/weights/best.pt')")
    print("results = model('path/to/your/image.jpg')")

if __name__ == "__main__":
    main()