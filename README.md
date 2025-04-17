# Multimodal Business Forecasting Project

This project explores advanced machine learning techniques for analyzing and forecasting business trends using **three different types of inputs**:

1. **Numerical Data**: Walmart sales data, including variables like weekly sales, temperature, fuel price, CPI, and unemployment rates.
2. **Image Data**: Product images sourced from e-commerce platforms and processed using CLIP to extract feature embeddings.
3. **Text Data**: Amazon Beauty product reviews, preprocessed using NLP techniques for sentiment and keyword analysis.

The goal is to predict future business trends by analyzing these diverse data types, providing strategic insights through AI-powered modeling and visualizations.

## Features
- **Multimodal Data Integration**: Combines structured numerical data, visual patterns, and unstructured text in a unified modeling pipeline.
- **Feature Engineering and Selection**: Uses techniques like Recursive Feature Elimination (RFE) and LASSO regression for identifying relevant features.
- **Predictive Modeling**:
  - **Tabular Models**: Linear Regression, Decision Trees, Random Forest, XGBoost, LightGBM, CatBoost, SVM, and Neural Networks (MLP).
  - **Image Models**: CLIP embeddings processed through CNNs, ResNet, EfficientNet, MobileNet, and Vision Transformers (ViT).
  - **Text Models**: NLP pipelines using BERT, LSTM for sentiment analysis, LDA/NMF for topic modeling, and transformer-based text classifiers.
- **Model Optimization**: GridSearchCV and RandomizedSearchCV for hyperparameter tuning across models.
- **Evaluation**: R² Score, MSE, Precision, Recall, F1-score, Top-K Accuracy, BLEU Score, and others depending on task.

## Usage
1. Prepare your data with the required three input formats (tabular CSV, image folder or URLs, and textual reviews).
2. Configure the model pipeline and parameters in `pipeline.py`.
3. Run the main script:
   ```bash
   python main.py
   ```
4. Results including metrics and predictions will be saved in the `results/` folder.

## Requirements
Install required packages using pip:
```bash
pip install -r requirements.txt
```

## Output
The output includes:
- Feature selection reports
- Model evaluation metrics
- Prediction results for downstream business analysis

## Applications
This project is ideal for business intelligence tasks such as:
- Product demand forecasting
- Sales trend analysis
- Visual merchandising performance evaluation

## Notes
- Ensure all image data is preprocessed and aligned with the corresponding tabular and textual entries.
- Designed to be modular for easy integration of new models or modalities.