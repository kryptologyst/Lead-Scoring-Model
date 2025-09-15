# Lead Scoring Model

A comprehensive machine learning-powered lead scoring system that predicts the likelihood of lead conversion using behavioral and demographic data.

## Overview

This project implements a lead scoring model that helps sales and marketing teams prioritize leads based on their conversion probability. The system uses machine learning algorithms to analyze various factors including demographics, engagement metrics, and behavioral patterns to assign scores from 0-100.

## Features

- **Machine Learning Model**: Random Forest classifier with cross-validation
- **Web Interface**: Modern Flask-based UI for easy lead management
- **Real-time Scoring**: Instant lead score calculation with actionable insights
- **Analytics Dashboard**: Comprehensive model performance metrics and visualizations
- **Data Export**: CSV export functionality for external analysis
- **Mock Database**: Realistic synthetic data for testing and demonstration
- **Model Persistence**: Automatic model saving and loading
- **Responsive Design**: Mobile-friendly Bootstrap interface

## Project Structure

```
0057_Lead_scoring_model/
├── app.py                 # Flask web application
├── 0057.py              # Original simple implementation
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── .gitignore           # Git ignore rules
├── data/
│   └── mock_database.py # Mock database with synthetic lead data
├── models/
│   └── lead_scorer.py   # Core ML model implementation
└── templates/           # HTML templates
    ├── base.html        # Base template
    ├── index.html       # Dashboard
    ├── score_lead.html  # Lead scoring form
    ├── score_result.html # Scoring results
    ├── leads.html       # Lead management
    └── analytics.html   # Analytics dashboard
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   cd 0057_Lead_scoring_model
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:5000`

## Lead Scoring Categories

The system categorizes leads into four categories based on their scores:

- **🔥 Hot (80-100)**: High conversion probability - immediate follow-up recommended
- **🌡️ Warm (60-79)**: Good conversion potential - targeted nurturing needed
- **❄️ Cold (40-59)**: Moderate interest - continue with regular nurturing
- **🧊 Ice Cold (0-39)**: Low conversion likelihood - long-term nurturing

## Features Used for Scoring

### Demographics
- **Age**: Contact person's age
- **Industry**: Business sector (Technology, Healthcare, Finance, etc.)
- **Company Size**: From startup to enterprise

### Lead Source
- **Email**: Direct email campaigns
- **Website**: Organic website visits
- **Social Media**: Social platform referrals
- **Referral**: Word-of-mouth recommendations
- **Paid Ad**: Advertising campaigns
- **Webinar**: Educational events
- **Trade Show**: Industry events
- **Cold Call**: Outbound sales calls

### Engagement Metrics
- **Page Views**: Number of website pages visited
- **Time on Site**: Duration of website sessions
- **Email Opens**: Email engagement rate
- **Email Clicks**: Click-through behavior
- **Form Submissions**: Contact form completions
- **Content Downloads**: Resource downloads
- **Days Since Contact**: Recency of interaction

## Machine Learning Model

### Algorithm
- **Primary**: Random Forest Classifier
- **Alternatives**: Gradient Boosting, Logistic Regression
- **Preprocessing**: StandardScaler for numerical features, OneHotEncoder for categorical
- **Validation**: 5-fold cross-validation

### Performance Metrics
- **Accuracy**: Overall prediction accuracy
- **ROC AUC**: Area under the ROC curve
- **Precision/Recall**: Class-specific performance
- **Confusion Matrix**: Detailed classification results

### Model Features
- **Feature Importance**: Identifies most influential factors
- **Model Persistence**: Automatic saving/loading
- **Retraining**: Easy model updates with new data
- **Cross-validation**: Robust performance estimation

## Web Interface

### Dashboard
- Overview statistics and key metrics
- Quick action buttons for common tasks
- Model status and lead score categories
- System information and guidance

### Lead Scoring
- Intuitive form for entering lead information
- Real-time validation and helpful tips
- Instant score calculation with recommendations
- Detailed prediction explanations

### Lead Management
- Comprehensive lead listing with search
- Score-based filtering and sorting
- Export functionality for external tools
- Visual score categorization

### Analytics
- Model performance metrics
- Feature importance visualization
- Lead distribution charts
- Confusion matrix analysis
- Retraining capabilities

## API Endpoints

### REST API
```
POST /api/score_lead
Content-Type: application/json

{
  "lead_source": "Email",
  "industry": "Technology",
  "company_size": "Medium (51-200)",
  "age": 35,
  "page_views": 10,
  "time_on_site": 800,
  "email_opens": 5,
  "email_clicks": 2,
  "form_submissions": 1,
  "content_downloads": 2,
  "days_since_contact": 7
}
```

### Response
```json
{
  "success": true,
  "lead_score": 72,
  "conversion_probability": 0.72,
  "predicted_conversion": true,
  "score_category": "Warm"
}
```

## 🔧 Configuration

### Model Parameters
Edit `models/lead_scorer.py` to adjust:
- Algorithm selection (`model_type`)
- Feature engineering
- Hyperparameters
- Validation strategy

### Database
The mock database generates 1000 realistic leads by default. Modify `data/mock_database.py` to:
- Change sample size
- Adjust feature distributions
- Modify conversion probabilities
- Add new features

## Sample Data

The system includes a sophisticated mock database that generates realistic lead data with:
- **Correlated Features**: Realistic relationships between variables
- **Industry Variations**: Different conversion rates by sector
- **Source Quality**: Varying lead quality by acquisition channel
- **Behavioral Patterns**: Realistic engagement distributions
- **Temporal Factors**: Time-based conversion patterns

## Testing

### Manual Testing
1. Start the application
2. Navigate to "Score Lead"
3. Enter test data with various combinations
4. Verify score calculations and categories
5. Check analytics dashboard updates

### Model Testing
```bash
python models/lead_scorer.py
```

### Database Testing
```bash
python data/mock_database.py
```

## Deployment

### Local Development
- Use the built-in Flask development server
- Set `debug=True` for development features
- Access at `http://localhost:5000`

### Production Deployment
1. Set `debug=False` in `app.py`
2. Use a production WSGI server (Gunicorn, uWSGI)
3. Configure environment variables for secrets
4. Set up proper logging and monitoring
5. Use a reverse proxy (Nginx) for static files

### Docker Deployment
Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## Future Enhancements

### Technical Improvements
- [ ] Real database integration (PostgreSQL, MySQL)
- [ ] User authentication and authorization
- [ ] A/B testing framework for model comparison
- [ ] Real-time model monitoring and drift detection
- [ ] Advanced feature engineering pipeline
- [ ] Ensemble model implementation

### Business Features
- [ ] Lead assignment and routing
- [ ] Email integration for automated follow-ups
- [ ] CRM system integration
- [ ] Advanced reporting and dashboards
- [ ] Lead lifecycle tracking
- [ ] Conversion outcome feedback loop

### Technical Debt
- [ ] Comprehensive unit test suite
- [ ] Integration test coverage
- [ ] Performance optimization
- [ ] Security hardening
- [ ] API rate limiting
- [ ] Caching implementation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For questions, issues, or contributions:
- Create an issue in the repository
- Review the code documentation
- Check the analytics dashboard for model insights

## Learning Objectives

This project demonstrates:
- **Machine Learning**: Classification, feature engineering, model evaluation
- **Web Development**: Flask, HTML/CSS, JavaScript, Bootstrap
- **Data Science**: Data generation, analysis, visualization
- **Software Engineering**: Project structure, documentation, version control
- **Business Intelligence**: Lead scoring, conversion optimization, analytics
