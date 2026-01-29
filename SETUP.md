# 🚀 Setup Guide

This guide will help you set up and run the Solar Panel Classifier project.

## 📋 Prerequisites

- **Python:** 3.8 or higher
- **pip:** Latest version
- **Git:** For cloning the repository
- **(Optional) GPU:** NVIDIA GPU with CUDA for faster training

## 🔧 Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/solar-panel-classifier.git
cd solar-panel-classifier
```

### 2. Create Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download or Train Model

**Option A: Download Pre-trained Model**
1. Download the model file from [releases](https://github.com/yourusername/solar-panel-classifier/releases)
2. Place it in `models/solar_panel_efficientnet_optimized.h5`

**Option B: Train Your Own Model**
1. Prepare your dataset in the `Data/` directory
2. Open the training notebook: `notebooks/solar_panel_classification.ipynb`
3. Run all cells to train the model

### 5. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🗂️ Project Structure Setup

Create the following directory structure:

```
solar-panel-classifier/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── models/
│   └── (place your .h5 model file here)
├── notebooks/
│   └── solar_panel_classification.ipynb
├── data/
│   └── (prediction history will be saved here)
└── Data/
    ├── Bird-drop/
    ├── Clean/
    ├── Dusty/
    ├── Electrical-damage/
    ├── Physical-Damage/
    └── Snow-Covered/
```

## 🐛 Troubleshooting

### Issue: TensorFlow Installation Error

**Solution:**
```bash
pip install tensorflow --upgrade
```

For GPU support:
```bash
pip install tensorflow-gpu
```

### Issue: Streamlit Not Found

**Solution:**
```bash
pip install streamlit --upgrade
```

### Issue: Model File Not Found

**Solution:**
1. Check that your model file is in `models/` directory
2. Verify the filename matches: `solar_panel_efficientnet_optimized.h5`
3. Update `CONFIG['model_path']` in `app.py` if using different filename

### Issue: Memory Error During Training

**Solution:**
- Reduce batch size in the notebook
- Close other applications
- Use a machine with more RAM
- Use Google Colab for free GPU access

## 🌐 Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy!

**Note:** You may need to use Git LFS for large model files or host them separately.

### Deploy to Heroku

1. Install Heroku CLI
2. Create `Procfile`:
   ```
   web: streamlit run app.py --server.port $PORT
   ```
3. Deploy:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

## 📊 Using Kaggle for Training

1. Upload your notebook to Kaggle
2. Upload your dataset as a Kaggle dataset
3. Enable GPU accelerator in notebook settings
4. Run the notebook
5. Download the trained model

## 💡 Tips

- Use a GPU for training (much faster!)
- Start with the pre-trained model for quick testing
- Customize the UI colors in `app.py` CSS section
- Add your own sample images for testing
- Monitor training with TensorBoard

## 📧 Support

If you encounter issues:
1. Check the [FAQ](docs/faq.md)
2. Search [existing issues](https://github.com/yourusername/solar-panel-classifier/issues)
3. Create a new issue with detailed information

## 🎓 Learning Resources

- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)

---

**Happy Coding! 🚀**
