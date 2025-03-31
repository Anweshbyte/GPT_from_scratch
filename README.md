# 🚀 Transformer-Based Model Training & Evaluation

![Alt text](assets/image.png)

This repository provides a **production-ready** implementation of a transformer-based model for text generation.  
It includes **training, evaluation, and visualization** functionalities, making it easy to fine-tune and test models.

## 📌 Features
- ✅ Train a **GPT-style** model using PyTorch
- ✅ Evaluate loss on training and validation sets
- ✅ Generate sample text outputs after training
- ✅ Visualize loss curves with Matplotlib
- ✅ Well-structured for **production-ready deployment**

---

## 📂 Project Structure
```plaintext
📦 project-root
├── 📂 data              # Dataset files or scripts to download data
│   ├── dataset.py       # Dataset handling and preprocessing
│   ├── sample_data/     # Sample datasets for testing
│
├── 📂 models            # Model definitions
│   ├── gpt_model.py     # Transformer-based GPT model implementation
│
├── 📂 training          # Training and evaluation scripts
│   ├── train.py         # Main script for training the model
│   ├── evaluate.py      # Evaluation and inference script
│
├── 📂 utils             # Utility functions
│   ├── helpers.py       # Helper functions for tokenization, etc.
│
├── 📂 outputs           # Generated outputs (e.g., logs, sample texts)
│
├── requirements.txt     # Required dependencies
├── README.md            # Project documentation
├── config.yaml          # Model and training configurations
```

## 📥 Installation

1️⃣ Clone the repository
```bash
git clone https://github.com/Anweshbyte/GPT_from_scratch.git
cd GPT_from_scratch
```

2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Usage

To run the model (trained on "verdict.tct"), run:
```bash
cd flask_gpt_webapp
python app.py
```
Search http://127.0.0.1:5000 on the browser.

### 🔹 Training on Custom Dataset

To train on a custom txt file, simply change the "file_path" in "main.py"
```python
file_path = "/Users/arpitasen/Desktop/GPT/archive/merged.txt"

```
