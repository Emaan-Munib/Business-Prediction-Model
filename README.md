
# AI-Powered Business Time Machine
### Multimodal Business Forecasting & AI-Powered Analytics

A machine learning application exploring business intelligence across **structured sales data, customer reviews, and product images**.

The project develops Walmart weekly-sales prediction models, analyzes Amazon Beauty review sentiment, extracts visual representations using CLIP, and experiments with generative AI for explanations and product imagery.

Interactive **Gradio** interfaces make sales prediction and review analysis accessible without writing code.

> **Implementation scope:** The project contains separate tabular, text, and image workflows. These are multimodal components, rather than a verified end-to-end fused forecasting model. Some image-classification and generative-AI sections are experimental.

---

## Key Features

- **Sales Forecasting:** Train and compare seven regression algorithms on Walmart weekly sales, with an additional dense neural-network experiment.
- **Feature Selection:** Combine Recursive Feature Elimination (RFE) with a Decision Tree estimator and LassoCV.
- **Model Optimization:** Use GridSearchCV for selected regressors and experiment with RandomizedSearchCV.
- **Customer Sentiment Analysis:** Clean and lemmatize Amazon Beauty review text; classify reviews using a pretrained Hugging Face sentiment pipeline.
- **Visual Feature Extraction:** Download product images and extract CLIP ViT-L/14 embeddings for downstream experiments.
- **Image Classification Experiments:** Compare neural architectures applied to CLIP embedding vectors, including CNN and ResNet-like, EfficientNet-like, and MobileNet-like networks.
- **Generative AI Experiments:** Explore GPT-2-based prediction explanations and Stable Diffusion text-to-image generation.
- **Interactive Interfaces:** Gradio sales predictor and review sentiment analyzer, alongside notebook-based prediction controls using ipywidgets.

---

## Data Sources

| Modality | Dataset | Purpose |
|---|---|---|
| Tabular | Walmart Weekly Sales (`Walmart.csv`) | Predict weekly sales using store, holiday, weather, and economic inputs |
| Text | Amazon Reviews 2023 — All Beauty | Customer sentiment analysis and NLP preprocessing |
| Image | E-commerce product images | CLIP embeddings and image classification experiments |

The Walmart dataset contains **6,435 records and 8 columns**: `Store`, `Date`, `Weekly_Sales`, `Holiday_Flag`, `Temperature`, `Fuel_Price`, `CPI`, and `Unemployment`.

Amazon Reviews 2023 is available through [McAuley Lab on Hugging Face](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).

The image pipeline processes a subset of 50 images from the supplied image URL list.

---

## System Architecture

```text
                    Business Data Sources
                    /        |          \
             Walmart CSV  Amazon Reviews  Product Images
                  |            |                |
             Preprocessing  NLP Cleaning    CLIP ViT-L/14
                  |            |                |
           Feature Selection Sentiment Model Image Embeddings
                  |            |                |
          Regression Models Gradio Interface Neural Experiments
                  |
             Evaluation
                  |
             XGBoost Model
                  |
           Saved Model (joblib)
                  |
            Gradio Sales UI

       Additional Generative AI Experiments:
          GPT-2 + Stable Diffusion
```

---

## 1. Walmart Sales Prediction

The tabular pipeline explores sales trends, feature distributions, correlations, and model performance.

### Data Preprocessing

- Date conversion and missing-value inspection
- Feature normalization using MinMaxScaler
- Exploratory data analysis and correlation visualizations
- Outlier analysis using the Interquartile Range (IQR)
- Feature selection using RFE and LassoCV

### Machine Learning Models

Seven regression algorithms were evaluated:

1. Linear Regression
2. Random Forest Regressor
3. Decision Tree Regressor
4. Support Vector Regression (SVR)
5. CatBoost Regressor
6. XGBoost Regressor
7. LightGBM Regressor

A Keras-based dense neural network was also explored separately.

Selected models were optimized using **GridSearchCV**, with three-fold cross-validation.

### Model Evaluation Results

| Model | R² Score |
|---|---:|
| **XGBoost** | **0.9183** |
| Random Forest | 0.9182 |
| Decision Tree | 0.9182 |
| CatBoost | 0.9182 |
| LightGBM | 0.9182 |
| Support Vector Regression | 0.7982 |
| Linear Regression | 0.1175 |

**Best-performing model: XGBoost**, achieving an R² score of 0.9183 in the reported feature-selection experiment.

The feature-selection experiment selected `Store` as its predictor. A separate XGBoost model was subsequently trained using six input features for interactive sales prediction. Its performance should be evaluated independently.

### Interactive Sales Predictor

The application allows users to enter:

- Store ID
- Holiday Flag
- Temperature
- Fuel Price
- Consumer Price Index (CPI)
- Unemployment Rate

The trained XGBoost model generates an estimated weekly sales value.

The model is saved using `joblib` as `xgb_walmart_model.pkl`.

The project includes command-line predictions, Jupyter widgets, and Gradio-based interfaces.

---

## 2. Amazon Customer Review Sentiment Analysis

The NLP component analyzes customer feedback from the **Amazon Reviews 2023 — All Beauty** dataset.

### Text Processing

The notebook explores:

- Text cleaning and normalization
- Tokenization
- Stop-word removal
- Lemmatization
- Word cloud visualization
- Sentiment classification

### Sentiment Model

The sentiment analysis pipeline uses Hugging Face Transformers.

The default pretrained model resolved to:

`distilbert/distilbert-base-uncased-finetuned-sst-2-english`

A sample of **500 customer reviews** was analyzed.

| Predicted Sentiment | Number of Reviews |
|---|---:|
| Positive | 329 |
| Negative | 171 |
| Total | 500 |

These figures represent model predictions, not independently verified classification accuracy.

### Interactive Sentiment Analysis

The Gradio review analyzer allows users to enter review text and receive a predicted sentiment with a confidence score.

This component demonstrates how NLP can support customer-feedback monitoring and business analysis.

---

## 3. Product Image Analysis Using CLIP

The image-processing component explores visual representations of e-commerce product images.

### CLIP Feature Extraction

The project uses:

`openai/clip-vit-large-patch14`

The workflow includes:

1. Loading product image URLs
2. Downloading a subset of images
3. Preprocessing images
4. Extracting CLIP embeddings
5. Saving embeddings for downstream experiments

Extracted features are exported to:

`image_embeddings.csv`

### Deep Learning Experiments

Additional PyTorch experiments explore:

- 1D Convolutional Neural Networks
- ResNet-like architectures
- EfficientNet-like architectures
- MobileNet-like architectures

These are custom embedding-based networks inspired by established architectures, rather than necessarily using pretrained image-classification backbones.

The notebook uses synthetic labels in its image classification experiments. Consequently, these experiments demonstrate model implementation rather than validated product-category classification.

---

## 4. Generative AI Components

### GPT-2 Explanation Generation

The project explores GPT-2 for generating natural-language descriptions of prediction results.

The notebook includes an experimental workflow for exporting explanations to:

`model_explanations.csv`

These generated descriptions provide readable output but should not be treated as validated feature-attribution explanations.

### Stable Diffusion Image Generation

The project also explores Stable Diffusion v1.5 for text-to-image generation.

This component demonstrates how generative AI can produce product-style imagery from descriptive prompts.

The generative AI components are exploratory notebook experiments and are not integrated into the Gradio sales predictor.

---

## Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Sweetviz |
| Machine Learning | Scikit-learn, XGBoost, LightGBM, CatBoost |
| Deep Learning | PyTorch, TensorFlow, Keras |
| NLP | NLTK, Hugging Face Transformers |
| Computer Vision | CLIP, PIL |
| Generative AI | GPT-2, Stable Diffusion, Diffusers |
| User Interfaces | Gradio, ipywidgets |
| Model Serialization | Joblib |
| Development Environment | Jupyter Notebook, Google Colab |

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Emaan-Munib/Business-Prediction-Model.git
cd Business-Prediction-Model
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn \
xgboost lightgbm catboost nltk wordcloud datasets \
transformers torch torchvision tensorflow pillow \
requests tqdm joblib gradio ipywidgets jsonlines \
sweetviz diffusers
```

### 3. Prepare Datasets

Download the required datasets and update their paths in the notebook:

- Walmart Weekly Sales CSV
- Amazon Beauty customer reviews
- Product image URLs for the image-processing experiments

### 4. Run the Notebook

Open `Business_Prediction_Model.ipynb` in Jupyter Notebook or Google Colab.

Run the relevant cells in order to perform preprocessing, train models, evaluate results, and launch the interfaces.

For the sales interface, the trained XGBoost model must first be saved as `xgb_walmart_model.pkl`.

### 5. Launch Interactive Applications

Execute the relevant Gradio cells in the notebook and open the URL provided by Gradio.

**Note:** The current notebook contains exploratory sections, Google Colab-specific file paths, and external model dependencies. Some sections may require additional setup or debugging. It is not yet packaged as a verified one-command application.

---

## Project Outputs

The project produces or explores:

- Walmart sales trend visualizations
- Feature selection results
- Regression model comparison metrics
- Cross-validation results
- Actual-versus-predicted sales plots
- Saved XGBoost sales prediction model
- Customer review sentiment predictions
- CLIP image embeddings
- Experimental GPT-2-generated explanations
- Stable Diffusion-generated example imagery
- Interactive Gradio interfaces

---

## Limitations and Future Improvements

Potential improvements include:

- Preventing data leakage by incorporating preprocessing and feature selection within training pipelines
- Evaluating future sales predictions using chronological train-test splits
- Independently validating the final six-feature XGBoost model
- Using genuine annotated image datasets for classification
- Integrating modalities through aligned business entities and timestamps
- Developing a unified multimodal forecasting pipeline
- Refactoring the notebook into reusable Python modules
- Providing standalone application entry points
- Adding explainability methods such as SHAP
- Deploying the application as an accessible web service

---

## Potential Applications

- **Retail Analytics:** Analyze sales trends and store performance.
- **Demand Forecasting:** Estimate sales using historical and economic indicators.
- **Customer Experience Analysis:** Extract sentiment from product reviews.
- **Product Image Analytics:** Explore visual product representations.
- **Business Intelligence:** Combine insights from structured and unstructured business data.
- **Generative AI:** Explore automated descriptions and product imagery.

---

## Project Status

**Academic AI/ML Development Project**

The project demonstrates machine learning, NLP, computer vision, generative AI, and interactive application development.

Its components are documented according to their implementation and evaluation status, with opportunities for further integration into a unified business intelligence platform.

---

## References and Acknowledgments

- [Amazon Reviews 2023 — McAuley Lab](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Gradio](https://www.gradio.app/)
- [Scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
