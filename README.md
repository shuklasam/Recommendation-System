# AI-Powered Recommendation System 🎯

An intelligent web application that leverages machine learning to deliver personalized product recommendations. This system combines collaborative filtering, statistical ranking methods, and classification algorithms to enhance user satisfaction through precise, data-driven recommendations.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Technologies Used](#technologies-used)
- [Key Algorithms](#key-algorithms)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Model Performance](#model-performance)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This recommendation system is designed to provide users with highly relevant product suggestions by analyzing user behavior, product characteristics, and rating patterns. The system employs a hybrid approach combining multiple machine learning techniques to deliver accurate and personalized recommendations.

### Key Objectives

- **Enhance User Satisfaction**: Deliver personalized recommendations that match user preferences
- **Precise Product Ranking**: Utilize Wilson's confidence interval for statistically sound rankings
- **Intelligent Classification**: Categorize diverse products using Random Forest for improved recommendation accuracy
- **Scalable Architecture**: Built with modern web technologies for seamless user experience

## ✨ Features

### Core Functionality

- **🤖 AI-Powered Recommendations**: Machine learning-driven product suggestions
- **📊 Collaborative Filtering**: Leverages user behavior patterns and similarities
- **🎲 Wilson's Interval Scoring**: Statistical ranking method for reliable product ratings
- **🌲 Random Forest Classification**: Intelligent product categorization and classification
- **👥 User-Centric Design**: Personalized experience based on individual preferences
- **📈 Real-time Updates**: Dynamic recommendations that adapt to user interactions
- **🔍 Multi-criteria Filtering**: Advanced filtering based on categories, ratings, and user history

### Additional Features

- User preference learning and adaptation
- Cold-start problem handling for new users/products
- Explainable recommendations (why this product was suggested)
- A/B testing capabilities for model optimization
- Analytics dashboard for monitoring system performance

## 🏗️ System Architecture

```
┌─────────────────┐
│   Web Frontend  │
│   (User Interface)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Backend API    │
│  (Flask/Django) │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────┐ ┌──────────────┐
│Database │ │ ML Models    │
│ (SQLite/│ │ - Collaborative│
│  MySQL) │ │ - Random Forest│
└─────────┘ │ - Wilson Score│
            └──────────────┘
```

## 🛠️ Technologies Used

### Backend & Machine Learning

- **Python 3.8+**: Core programming language
- **Scikit-learn**: Machine learning algorithms and utilities
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Jupyter Notebook**: Model development and experimentation

### Web Framework

- **Flask/Django**: Web application framework (specify which you used)
- **RESTful API**: For client-server communication

### Data & Storage

- **SQLite/PostgreSQL**: Database for user and product data
- **Pickle/Joblib**: Model serialization and deployment

### Development Tools

- **Git**: Version control
- **Virtual Environment**: Dependency isolation

## 🔬 Key Algorithms

### 1. Collaborative Filtering

Analyzes user-item interactions to find patterns and similarities between users or items.

**Types Implemented:**
- **User-Based Filtering**: Recommends items liked by similar users
- **Item-Based Filtering**: Suggests items similar to those the user liked

```python
# User similarity calculation using cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

user_similarity = cosine_similarity(user_item_matrix)
```

### 2. Wilson's Confidence Interval

A statistical method for ranking products based on ratings, providing more reliable scores than simple averages by accounting for the number of ratings and their distribution.

**Formula:**
```
Wilson Score = (positive + 1.9208) / (positive + negative) - 
               1.96 * sqrt((positive * negative) / (positive + negative) + 0.9604) / 
               (positive + negative) / (1 + 3.8416 / (positive + negative))
```

**Benefits:**
- Prevents highly-rated items with few reviews from dominating rankings
- Provides confidence intervals for rating reliability
- Statistically sound approach to ranking

### 3. Random Forest Classification

Ensemble learning method used to classify products into categories and predict user preferences.

**Applications in this system:**
- Product categorization
- User preference classification
- Feature importance analysis for recommendations

```python
from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/recommendation-system.git
cd recommendation-system
```

2. **Create and activate virtual environment:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up the database:**
```bash
python setup_database.py
```

5. **Train the initial models:**
```bash
python train_models.py
```

6. **Run the application:**
```bash
# For Flask
python app.py

# For Django
python manage.py runserver
```

7. **Access the application:**
Open your browser and navigate to `http://localhost:5000` (or the port specified)

### Dependencies

Create a `requirements.txt` file with:

```txt
flask==2.3.0  # or django==4.2.0
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.2.0
scipy==1.10.0
matplotlib==3.7.0
seaborn==0.12.0
jupyter==1.0.0
sqlalchemy==2.0.0
joblib==1.2.0
```

## 🚀 Usage

### For End Users

1. **Register/Login**: Create an account or log in
2. **Browse Products**: Explore the product catalog
3. **Rate Products**: Rate items you've interacted with
4. **Get Recommendations**: View personalized suggestions on your dashboard
5. **Explore Categories**: Filter recommendations by category

### For Developers

#### Training the Model

```python
from src.models.collaborative_filtering import CollaborativeFilteringModel
from src.models.random_forest_classifier import ProductClassifier

# Train collaborative filtering model
cf_model = CollaborativeFilteringModel()
cf_model.train(user_item_matrix)
cf_model.save('models/collaborative_filtering.pkl')

# Train product classifier
classifier = ProductClassifier()
classifier.train(product_features, product_categories)
classifier.save('models/product_classifier.pkl')
```

#### Making Recommendations

```python
from src.recommendation_engine import RecommendationEngine

engine = RecommendationEngine()

# Get recommendations for a user
recommendations = engine.get_recommendations(
    user_id=123,
    n_recommendations=10,
    method='hybrid'  # 'collaborative', 'content-based', or 'hybrid'
)

print(recommendations)
```

#### Calculating Wilson Score

```python
from src.utils.ranking import wilson_score

# Calculate Wilson score for a product
score = wilson_score(
    positive_ratings=85,
    total_ratings=100,
    confidence=0.95
)

print(f"Wilson Score: {score:.4f}")
```

## 📁 Project Structure

```
recommendation-system/
│
├── app.py                          # Main application entry point
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── data/
│   ├── raw/                        # Raw data files
│   │   ├── products.csv
│   │   ├── users.csv
│   │   └── ratings.csv
│   ├── processed/                  # Processed data
│   └── sample/                     # Sample datasets for testing
│
├── models/
│   ├── collaborative_filtering.pkl # Trained CF model
│   ├── product_classifier.pkl      # Random Forest classifier
│   └── scaler.pkl                  # Feature scaler
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_collaborative_filtering.ipynb
│   ├── 03_wilson_scoring.ipynb
│   ├── 04_random_forest_classification.ipynb
│   └── 05_model_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py          # Data loading utilities
│   │   └── preprocessor.py         # Data preprocessing
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── collaborative_filtering.py
│   │   ├── random_forest_classifier.py
│   │   └── base_model.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── ranking.py              # Wilson score and ranking
│   │   ├── metrics.py              # Evaluation metrics
│   │   └── helpers.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py               # API endpoints
│   │   └── serializers.py
│   │
│   └── recommendation_engine.py    # Main recommendation logic
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── recommendations.html
│   └── product_detail.html
│
├── tests/
│   ├── test_models.py
│   ├── test_api.py
│   └── test_utils.py
│
├── config/
│   ├── config.py                   # Configuration settings
│   └── database.py                 # Database configuration
│
└── scripts/
    ├── train_models.py
    ├── setup_database.py
    └── generate_sample_data.py
```

## 📚 API Documentation

### Endpoints

#### Get Recommendations

```http
GET /api/recommendations/{user_id}
```

**Parameters:**
- `user_id` (integer): User identifier
- `n` (integer, optional): Number of recommendations (default: 10)
- `method` (string, optional): Recommendation method ('collaborative', 'content', 'hybrid')

**Response:**
```json
{
  "user_id": 123,
  "recommendations": [
    {
      "product_id": 456,
      "product_name": "Wireless Headphones",
      "category": "Electronics",
      "wilson_score": 0.87,
      "predicted_rating": 4.5,
      "confidence": 0.92
    }
  ],
  "method": "hybrid"
}
```

#### Rate Product

```http
POST /api/rate
```

**Request Body:**
```json
{
  "user_id": 123,
  "product_id": 456,
  "rating": 5
}
```

#### Get Product Classification

```http
GET /api/classify/{product_id}
```

**Response:**
```json
{
  "product_id": 456,
  "predicted_category": "Electronics",
  "confidence": 0.94,
  "top_features": ["wireless", "battery_life", "sound_quality"]
}
```

## 📊 Model Performance

### Collaborative Filtering Metrics

| Metric | Score |
|--------|-------|
| RMSE | X.XX |
| MAE | X.XX |
| Precision@10 | XX% |
| Recall@10 | XX% |
| Coverage | XX% |

### Random Forest Classification

| Metric | Score |
|--------|-------|
| Accuracy | XX% |
| Precision | XX% |
| Recall | XX% |
| F1-Score | XX% |

### Wilson Score Impact

- **Ranking Stability**: XX% improvement in ranking consistency
- **User Satisfaction**: XX% increase in user engagement
- **Cold Start Handling**: XX% better performance for new products

## 📸 Screenshots

### Home Page
![Home Page](screenshots/home.png)

### Recommendations Dashboard
![Recommendations](screenshots/recommendations.png)

### Product Details
![Product Details](screenshots/product_detail.png)

## 🔮 Future Enhancements

### Planned Features

- [ ] Deep Learning integration (Neural Collaborative Filtering)
- [ ] Real-time recommendation updates using streaming data
- [ ] Multi-modal recommendations (text, images, metadata)
- [ ] Context-aware recommendations (time, location, device)
- [ ] Social features (friend recommendations, social proof)
- [ ] Advanced A/B testing framework
- [ ] Mobile application (iOS/Android)
- [ ] Integration with popular e-commerce platforms

### Research Directions

- Exploring transformer-based recommendation models
- Implementing federated learning for privacy-preserving recommendations
- Graph Neural Networks for relationship modeling
- Reinforcement learning for sequential recommendations

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Contribution Guidelines

- Write clear, commented code
- Follow PEP 8 style guide for Python
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**
- GitHub: Samarth Shukla (https://github.com/shuklasam/)
- Email: samritik2000@gmail.com

## 🙏 Acknowledgments

- Dataset sources and contributors
- Scikit-learn and Python data science community
- Research papers on recommendation systems
- Wilson's confidence interval research by Edwin B. Wilson
- Open-source contributors

## 📖 References

1. Wilson, E. B. (1927). "Probable Inference, the Law of Succession, and Statistical Inference"
2. Koren, Y., Bell, R., & Volinsky, C. (2009). "Matrix Factorization Techniques for Recommender Systems"
3. Breiman, L. (2001). "Random Forests"
4. Ricci, F., Rokach, L., & Shapira, B. (2015). "Recommender Systems Handbook"

---

⭐ **If you find this project helpful, please consider giving it a star!**

**Happy Recommending!** 🎯
