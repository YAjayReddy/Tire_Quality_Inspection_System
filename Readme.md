

```markdown
# ML Model for Tire Quality Inspection

## Description

The ML Model for Tire Quality Inspection is designed to automate the identification and classification of defects in tires, aiming to improve the quality control process in the tire manufacturing industry. This project utilizes deep learning techniques to detect various defects, such as unwanted stains, demolding traces, foreign substances, air bubbles, and shape shifting, ensuring higher quality and efficiency in tire production.

### **Project Overview**

- **Aim**: To develop a machine learning model that classifies different tire defects, identifies tire colors, and performs optical character recognition (OCR) on tire codes during the manufacturing process.
- **Methods**:
    - **Tire/Camera Data Acquisition**: Stereo-photometric cameras capture high-resolution images of tire surfaces across different zones.
    - **Deep Learning Techniques**:
      1. **Instance Segmentation**: Detect and localize defects within tire images.
      2. **Classification**: Classify defects into severity levels based on detected features.
      3. **Model Validation**: Achieved F1 scores between 0.7 and 0.89 across different tire zones, validating the model's effectiveness.
    
### **Benefits**
- **Higher Precision**: Faster and more accurate defect detection than traditional methods.
- **Effective Quality Control**: Automates the inspection process, reducing labor costs and speeding up production.
- **Preventive Maintenance**: Predict potential defects before they occur, using historical data to forecast failures.

### **Challenges**
- **Data Diversity**: Requires a comprehensive dataset covering various tire types and defects.
- **Generalization Across Plants**: The model needs to be adaptable to different manufacturing environments, requiring ongoing tuning.

This project represents a step forward in the full automation of tire quality control, improving safety and performance in automotive industries.



## Installation Instructions

To set up and run the tire quality inspection model, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YAjayReddy/Tire_Quality_Inspection_System
   cd tire-quality-inspection
   ```

2. **Set up YOLOv5**:
   Follow the official YOLOv5 setup guide, or use the following command to install the required dependencies:
   ```bash
   pip install -U yolov5
   ```

3. **Install Dependencies**:
   The model requires the following dependencies:
   - **PyTorch** (for deep learning model training)
   - **OpenCV** (for image processing)
   - **Tesseract** (for OCR)
   
   Install them with the following commands:
   ```bash
   pip install torch torchvision torchaudio
   pip install opencv-python
   pip install pytesseract
   ```

## Usage

After training the model, use the following commands to run it on new tire images:

1. **Running the Model**:
   To test the trained model on new images, run:
   ```bash
   python detect.py --weights yolov5s.pt --img-size 640 --source path_to_image_or_folder
   ```

2. **OCR on Tire Codes**:
   Use the following command to perform OCR on the tire code:
   ```bash
   python ocr.py --image path_to_image
   ```

This will output the detected defects and tire codes.

## Contributing

We welcome contributions to this project! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add new feature"`).
4. Push to your fork (`git push origin feature-branch`).
5. Create a pull request.

### **Development Setup**

Ensure you have Python 3.6+ installed, along with the following tools:
- PyCharm, VSCode, or your preferred IDE.
- Install dependencies using `pip install -r requirements.txt`.

Please follow PEP 8 standards for Python code style.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Contact Information

For questions or support, please contact:

- Email: ajayreddyyamala@gmail.com
- GitHub: https://github.com/YAjayReddy

## Acknowledgments

- **YOLOv5**: For object detection and defect localization.
- **Tesseract OCR**: For optical character recognition of tire codes.
- **OpenCV**: For image processing.
- Special thanks to the contributors and research papers that inspired this project.

## Badges

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

```

---

