 Housing Price Prediction with PyTorch
 A NN project that predicts housing prices using multilayer neural networks built with PyTorch. Three architectural configurations were compared across different activation functions, learning rates, and network depths.
**Problem description**
Predicting real estate prices based on structural and locational features is a classic regression problem with practical value in the housing market. This project explores how neural network design choices — activation functions, depth, dropout rate, and learning rate — affect regression performance on tabular housing data.

**Dataset**
The Housing Prices Dataset contains 545 records with 13 features including area, number of bedrooms/bathrooms/stories, and binary amenity indicators (mainroad, guestroom, basement, hot water heating, air conditioning, preferred area) plus furnishing status.
Source: Kaggle — https://www.kaggle.com/datasets/yasserh/housing-prices-dataset
Target variable: price (normalized to millions)
Split: 60% train / 20% val / 20% test
Preprocessing: Binary encoding for yes/no columns, ordinal encoding for furnishing status, StandardScaler normalization


## 🧪 Experiments & Results

| Experiment | Architecture | Activation | LR | Epochs | Train Loss | Val Loss | Test MSE | R² Score |
|------------|-------------|------------|------|--------|-----------|---------|---------|---------|
| Exp 1 | [64, 32] | ReLU | 0.01 | 200 | 0.8407 | 1.2769 | 2.1642 | 0.5718 |
| Exp 2 | [128, 64, 32] | Tanh | 0.001 | 200 | 13.4371 | 12.6950 | 14.8806 | −1.9440 |
| **Exp 3** | **[128, 64]** | **ReLU** | **0.003** | **300** | **0.3149** | **1.1495** | **1.9659** | **0.6111** |


Experiment 3 achieved the best performance with R² = 0.61 and the lowest Test MSE (1.97). ReLU with moderate dropout (0.1 / 0.05) and a well-tuned learning rate of 0.003 over 300 epochs outperformed both shallower and Tanh-based architectures.

**How to run**
1. Clone the repository
https://github.com/RahafEA/nnfinal
3. Install dependencies
pip install torch scikit-learn pandas matplotlib numpy
4. Download the dataset
Download Housing.csv from the Kaggle dataset page and place it in the project root, or update the path in the script:
df = pd.read_csv("Housing.csv")  
5. Run the training script
python housing_price_prediction.py

