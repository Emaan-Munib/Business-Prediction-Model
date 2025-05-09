# Multimodal Business Forecasting Project

This project leverages advanced machine learning and generative AI techniques to analyze and forecast business trends using **three distinct data modalities**:

1. **Numerical Data**: Walmart sales data, including variables like weekly sales, temperature, fuel price, CPI, and unemployment rates.
2. **Image Data**: Product images sourced from e-commerce platforms and processed using CLIP to extract feature embeddings.
3. **Text Data**: Amazon Beauty product reviews, preprocessed using NLP techniques for sentiment analysis and keyword extraction.

The goal is to predict future business trends by analyzing these diverse data types, providing strategic insights through AI-powered modeling, visualizations, and interpretive explanations.

---

## 🔥 **Key Features**

* **Multimodal Data Integration**: Combines structured numerical data, visual patterns, and unstructured text in a unified modeling pipeline.
* **Feature Engineering and Selection**: Utilizes techniques like Recursive Feature Elimination (RFE) and LASSO regression to identify relevant features.
* **Predictive Modeling**:

  * **Tabular Models**: Linear Regression, Decision Trees, Random Forest, XGBoost, LightGBM, CatBoost, SVM, and MLP Neural Networks.
  * **Image Models**: CLIP embeddings processed through CNNs, ResNet, EfficientNet, MobileNet, and Vision Transformers (ViT).
  * **Text Models**: NLP pipelines using BERT, LSTM for sentiment analysis, LDA/NMF for topic modeling, and transformer-based text classifiers.
* **Model Optimization**: GridSearchCV and RandomizedSearchCV for hyperparameter tuning across models.
* **Evaluation Metrics**: R² Score, MSE, Precision, Recall, F1-score, Top-K Accuracy, BLEU Score, and more depending on the task.
* **Generative AI for Explanations**: Uses GPT-2 to automatically generate human-readable explanations for model predictions, enhancing interpretability.
* **Image Generation from Prompts**: Leverages Stable Diffusion to create custom images based on textual prompts, enabling visual augmentation for business insights.

---

## 🚀 **Usage**

1. Prepare your data in the required formats:

   * **Tabular**: CSV format with all necessary features.
   * **Images**: Folder or URLs pointing to product images.
   * **Text**: JSON or CSV with product reviews or descriptions.

2. Configure the model pipeline and parameters in `pipeline.py`.

3. Run the main script:

   ```bash
   python main.py
   ```

4. Generated explanations and image visualizations are saved in the `results/` folder:

   * **model\_explanations.csv** for interpretive text
   * **generated\_images/** for AI-generated visuals

---

## 📌 **Requirements**

Install required packages using pip:

```bash
pip install -r requirements.txt
```

---

## 🎯 **Output**

The output includes:

* Feature selection reports
* Model evaluation metrics
* Prediction results for downstream business analysis
* Human-readable AI-generated explanations of model predictions
* AI-generated images based on custom prompts

---

## 💡 **Applications**

This project is ideal for business intelligence tasks such as:

* Product demand forecasting
* Sales trend analysis
* Visual merchandising performance evaluation
* Marketing strategy optimization through visual and textual insights

---

## 📝 **Notes**

* Ensure all image data is preprocessed and aligned with the corresponding tabular and textual entries.
* Generative AI features for explanation and image generation are optional but enhance interpretability and strategic insights.
* Designed to be modular for easy integration of new models or modalities.
