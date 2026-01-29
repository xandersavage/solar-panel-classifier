# ☀️ Solar Panel Condition Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**AI-powered solar panel defect detection using deep learning**

[Live Demo](#) | [Report Bug](https://github.com/yourusername/solar-panel-classifier/issues) | [Request Feature](https://github.com/yourusername/solar-panel-classifier/issues)

</div>

---

## 📋 Table of Contents

- [About The Project](#about-the-project)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## 🎯 About The Project

This project implements an **AI-powered solar panel inspection system** that can automatically detect and classify six different conditions of solar panels:

- ✅ **Clean** - Optimal condition
- 🦅 **Bird-drop** - Bird droppings contamination
- 💨 **Dusty** - Dust accumulation
- ⚡ **Electrical-damage** - Electrical malfunction
- 🔨 **Physical-Damage** - Physical damage to structure
- ❄️ **Snow-Covered** - Snow coverage

The system uses **transfer learning** with EfficientNetB0 architecture, achieving **95%+ accuracy** on the validation set.

### Built With

- [TensorFlow](https://www.tensorflow.org/) - Deep learning framework
- [Keras](https://keras.io/) - High-level neural networks API
- [Streamlit](https://streamlit.io/) - Web app framework
- [EfficientNet](https://arxiv.org/abs/1905.11946) - CNN architecture
- [Plotly](https://plotly.com/) - Interactive visualizations

---

## ✨ Features

- 🤖 **Deep Learning Model** - EfficientNetB0 with transfer learning
- 📊 **Interactive Dashboard** - Beautiful Streamlit web interface
- 📈 **Real-time Analysis** - Instant predictions with confidence scores
- 🎯 **High Accuracy** - 95%+ validation accuracy
- 💡 **Actionable Insights** - Recommendations for each condition
- 📉 **Visualization** - Interactive charts and confidence gauges
- 💾 **History Tracking** - Save and track prediction history
- 📱 **Responsive Design** - Works on desktop and mobile

---

## 📊 Dataset

The dataset consists of **885 images** across 6 classes:

| Class | Images | Description |
|-------|--------|-------------|
| Bird-drop | ~150 | Bird droppings on panels |
| Clean | ~150 | Clean, optimal condition |
| Dusty | ~150 | Dust-covered panels |
| Electrical-damage | ~150 | Electrical faults |
| Physical-Damage | ~135 | Physical damage/cracks |
| Snow-Covered | ~150 | Snow-covered panels |

**Dataset Split:**
- Training: 708 images (80%)
- Validation: 177 images (20%)

---

## 🏗️ Model Architecture

### Base Model: EfficientNetB0

- **Pre-trained on:** ImageNet
- **Input size:** 224x224x3
- **Parameters:** ~5.3M
- **Transfer Learning:** Last 20 layers unfrozen

### Custom Classification Head

```
EfficientNetB0 (frozen base)
    ↓
Global Average Pooling
    ↓
Dropout (0.3)
    ↓
Dense (128 units, ReLU)
    ↓
Batch Normalization
    ↓
Dropout (0.2)
    ↓
Dense (6 units, Softmax)
```

### Training Configuration

- **Optimizer:** Adam
- **Learning Rate:** 1e-4 (with reduction on plateau)
- **Loss:** Categorical Crossentropy
- **Batch Size:** 32
- **Epochs:** 25 (with early stopping)
- **Data Augmentation:**
  - Random horizontal flip
  - Random rotation (±20%)
  - Random zoom (±20%)
  - Random contrast adjustment
  - Random brightness adjustment

### Hyperparameter Optimization

Used **Keras Tuner (Hyperband)** to optimize:
- Number of unfrozen layers
- Dropout rates
- Dense layer units
- Learning rate

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) GPU with CUDA for faster training

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/solar-panel-classifier.git
   cd solar-panel-classifier
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the trained model**
   
   Place your trained model file in the `models/` directory:
   ```
   models/solar_panel_efficientnet_optimized.h5
   ```

---

## 💻 Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Training the Model

1. **Prepare your dataset** in the following structure:
   ```
   Data/
   ├── Bird-drop/
   ├── Clean/
   ├── Dusty/
   ├── Electrical-damage/
   ├── Physical-Damage/
   └── Snow-Covered/
   ```

2. **Open the Jupyter notebook**
   ```bash
   jupyter notebook notebooks/solar_panel_classification.ipynb
   ```

3. **Run all cells** to train the model

### Making Predictions Programmatically

```python
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model('models/solar_panel_efficientnet_optimized.h5')

# Load and preprocess image
image = Image.open('path/to/your/image.jpg').convert('RGB')
img = image.resize((224, 224))
img_array = np.array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array.astype(np.float32))

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class]

classes = ["Bird-drop", "Clean", "Dusty", "Electrical-damage", 
           "Physical-Damage", "Snow-Covered"]

print(f"Prediction: {classes[predicted_class]}")
print(f"Confidence: {confidence:.2%}")
```

---

## 📁 Project Structure

```
solar-panel-classifier/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── LICENSE                         # MIT License
├── .gitignore                      # Git ignore file
│
├── models/                         # Trained models
│   └── solar_panel_efficientnet_optimized.h5
│
├── notebooks/                      # Jupyter notebooks
│   ├── solar_panel_classification.ipynb
│   └── exploratory_data_analysis.ipynb
│
├── data/                          # Data directory
│   ├── prediction_history.json    # Prediction logs
│   └── sample_images/             # Sample test images
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── model.py                   # Model architecture
│   ├── preprocessing.py           # Data preprocessing
│   ├── utils.py                   # Utility functions
│   └── config.py                  # Configuration settings
│
├── tests/                         # Unit tests
│   ├── __init__.py
│   └── test_model.py
│
├── docs/                          # Documentation
│   ├── model_architecture.md
│   ├── training_guide.md
│   └── deployment_guide.md
│
└── assets/                        # Static assets
    ├── images/                    # Images for README
    └── demo/                      # Demo videos/gifs
```

---

## 📈 Results

### Model Performance

| Metric | Score |
|--------|-------|
| **Validation Accuracy** | 95.48% |
| **Validation Loss** | 0.1523 |
| **Precision** | 0.9521 |
| **Recall** | 0.9548 |
| **F1-Score** | 0.9534 |

### Per-Class Accuracy

| Class | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| Bird-drop | 96.7% | 0.95 | 0.97 |
| Clean | 98.3% | 0.98 | 0.98 |
| Dusty | 93.3% | 0.93 | 0.93 |
| Electrical-damage | 95.0% | 0.96 | 0.95 |
| Physical-Damage | 94.4% | 0.94 | 0.94 |
| Snow-Covered | 95.0% | 0.95 | 0.95 |

### Confusion Matrix

![Confusion Matrix](assets/images/confusion_matrix.png)

### Training History

![Training History](assets/images/training_history.png)

---

## 🗺️ Roadmap

- [x] Build baseline model with EfficientNetB0
- [x] Implement hyperparameter optimization
- [x] Create Streamlit web application
- [x] Add interactive visualizations
- [ ] Deploy to cloud (Streamlit Cloud/Heroku)
- [ ] Add batch processing capability
- [ ] Implement real-time video analysis
- [ ] Create mobile app version
- [ ] Add multi-language support
- [ ] Integrate with IoT sensors

See the [open issues](https://github.com/yourusername/solar-panel-classifier/issues) for a full list of proposed features.

---

## 🤝 Contributing

Contributions make the open-source community an amazing place to learn and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 📧 Contact

Alexander Olomukoro - [@yourtwitter](https://twitter.com/yourtwitter) - your.email@example.com

Project Link: [https://github.com/yourusername/solar-panel-classifier](https://github.com/yourusername/solar-panel-classifier)

---

## 🙏 Acknowledgments

- [EfficientNet Paper](https://arxiv.org/abs/1905.11946) - Model architecture
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials) - Learning resource
- [Streamlit Documentation](https://docs.streamlit.io/) - Web app framework
- [Kaggle](https://www.kaggle.com/) - Training platform
- Solar panel dataset contributors
- All contributors who helped improve this project

---

<div align="center">

**⭐ Star this repo if you find it helpful! ⭐**

Made with ❤️ by [Alexander Olomukoro](https://github.com/yourusername)

</div>
