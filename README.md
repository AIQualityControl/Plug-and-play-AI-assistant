# Plug-and-play AI Measurements and Diagnosis Assistant in Prenatal Ultrasound

## 📌 Overview

![Fetal Ultrasound AI Workflow](fetal_ai_overview.png)  <!-- 本地图片示例 -->

This repository contains an AI-powered assistant for automated fetal biometry measurement and abnormality detection in prenatal ultrasound examinations. The system helps address critical challenges in prenatal care by:

- Reducing operator variability in measurements
- Improving diagnostic accuracy for fetal abnormalities
- Providing standardized assessments comparable to senior sonographers

## ✨ Key Features

### 📏 Automated Biometric Measurements
Accurately measures 7 key fetal growth parameters:
- Head Circumference (HC)
- Biparietal Diameter (BPD)
- Transverse Cerebellar Diameter (TCD)
- Femur Length (FL)
- Humerus Length (HL)
- Abdominal Circumference (AC)
- Lateral Ventricular Width (LVW)

### 🚨 Abnormality Detection
Real-time diagnosis of 6 clinically significant conditions:
- Intrauterine Growth Restriction (IUGR)
- Microcephaly (Micro)
- Skeletal Dysplasia (SD)
- Hydrocephalus (Hyd)
- Congenital Heart defects (CH)
- Other fetal malformations

### 🏥 Clinical Integration
- Plug-and-play solution requiring no hardware modifications
- Seamless integration with existing PACS systems
- Real-time feedback during ultrasound examinations

## 📊 Performance Highlights
- **Strong agreement** with senior sonographers (ICC: 0.60-0.96 across parameters)
- **High diagnostic accuracy**: 84.44%-100% for abnormalities in external validation
- **Validated** in multi-center study with:
  - 45,117 training cases
  - 1,200 real-time scans
  - 4,396 retrospective cases

## Installation

### Prerequisites

- Python ≥ 3.6
- numpy>=1.17.2
- opencv-python>=4.2.0.34
- pynetdicom>=2.4.0
- PySide6
- pywin32
- loguru>=0.6.0
- pandas>=1.4.2
- scikit-learn>=1.4.0
- torch>=2.2.0
- torchvision>=0.17.0
- mmdet>=3.3.0
- mmcv>=2.1.0
- timm>=0.9.0
- imgaug>=0.4.0
- yacs>=0.1.8
### Install python env

To install required dependencies on the virtual environment of the python (e.g., virtualenv for python3), please run the following command at the root of this code:
```
$ python3 -m venv /path/to/new/virtual/environment/.
$ source /path/to/new/virtual/environment/bin/activate
```
For example:
```
$ mkdir python_env
$ python3 -m venv python_env/
$ source python_env/bin/activate
```
 

### Build Detectron2 from Source

Follow the [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) to install Detectron2.

## Dataset weight from Google Drive

Download the weight form https://drive.google.com/drive/folders/1HycEORHhoP_B9bDyzl-v9Xo6FMk3T6Nh



## Testing 

python measure_test.py 

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We thank the participating hospitals and sonographers who contributed data and expertise to this project.

## Code Reference 
  - [detectron2](https://github.com/facebookresearch/detectron2)
  - [YOLO](https://github.com/ultralytics/yolov5)
