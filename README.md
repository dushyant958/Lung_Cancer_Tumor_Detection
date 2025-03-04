# Medical Image Classification: Lung and Colon Tumor Detection

## Project Overview
This project implements a deep learning solution for medical image classification, specifically designed to distinguish between different types of lung and colon medical images. Utilizing Convolutional Neural Networks (CNN), the model aims to provide an automated approach to medical image analysis.

## Features
- **Advanced CNN architecture** for medical image classification
- **Robust preprocessing pipeline** for efficient data handling
- **Automated training** with early stopping
- **Comprehensive model evaluation metrics**
- **Visualization of training progress**

## Prerequisites
### Required Libraries
Ensure you have the following libraries installed:

```bash
numpy
pandas
matplotlib
scikit-learn
opencv-python
tensorflow
keras
Pillow
```

### System Requirements
- **Python 3.8+**
- **GPU recommended** (CUDA-enabled for TensorFlow)

## Installation

### Clone the repository:
```bash
git clone https://github.com/yourusername/medical-image-classification.git
cd medical-image-classification
```

### Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Install required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

### Data Structure
Ensure your dataset is organized as follows:
```plaintext
lung_colon_image_set/
│
├── lung/
│   ├── image1.jpeg
│   ├── image2.jpeg
│   └── ...
│
└── colon/
    ├── image1.jpeg
    ├── image2.jpeg
    └── ...
```

### Key Configuration
- Modify **DATASET_PATH** in the script to point to your dataset
- **Image size:** 256x256 pixels
- **Supported format:** JPEG

## Usage
To run the classification script, use:
```bash
python medical_image_classification.py
```

## Model Architecture
The neural network consists of:
- **3 Convolutional Layers**
- **Max Pooling Layers**
- **Batch Normalization**
- **Dropout for Regularization**
- **Softmax Output Layer**

## Training Strategies
- **Adam Optimizer**
- **Categorical Crossentropy Loss**
- **Early Stopping**
- **Learning Rate Reduction**
- **Automatic Training Termination at 90% Validation Accuracy**

## Visualization
The script generates:
- **Sample image previews**
- **Training loss and accuracy plots**
- **Confusion matrix**
- **Detailed classification report**

## Performance Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

## Troubleshooting
- Ensure correct path to dataset
- Check image format and size
- Verify GPU compatibility
- Install all dependencies

## Contributing
We welcome contributions! To contribute:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
Distributed under the **MIT License**. See `LICENSE` for more information.

## Contact
**Your Name** - dushyantatalkar415@gmail.com

**Project Link:** [GitHub Repository](https://github.com/dushyant958/Lung_Colon_Detection)

## Acknowledgements
This project utilizes the following technologies:
- **TensorFlow**
- **Keras**
- **scikit-learn**
- **OpenCV**
- **NumPy**
- **Pandas**

