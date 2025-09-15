"""
Flask Web Application for Lead Scoring Model
Provides a web interface for lead scoring and model management
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import json
import os
from datetime import datetime
from data.mock_database import MockLeadDatabase
from models.lead_scorer import LeadScoringModel

app = Flask(__name__)
app.secret_key = 'lead_scoring_secret_key_2024'

# Initialize database and model
db = MockLeadDatabase()
model = LeadScoringModel()

# Model file path
MODEL_PATH = 'models/trained_model.joblib'

# Load existing model if available
if os.path.exists(MODEL_PATH):
    try:
        model.load_model(MODEL_PATH)
        print("Loaded existing trained model")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Will train new model on first prediction")

@app.route('/')
def index():
    """Main dashboard"""
    stats = db.get_conversion_stats()
    return render_template('index.html', stats=stats, model_trained=model.is_trained)

@app.route('/score_lead', methods=['GET', 'POST'])
def score_lead():
    """Score a new lead"""
    if request.method == 'POST':
        try:
            # Get form data
            lead_data = {
                'lead_source': request.form['lead_source'],
                'industry': request.form['industry'],
                'company_size': request.form['company_size'],
                'age': int(request.form['age']),
                'page_views': int(request.form['page_views']),
                'time_on_site': int(request.form['time_on_site']),
                'email_opens': int(request.form['email_opens']),
                'email_clicks': int(request.form['email_clicks']),
                'form_submissions': int(request.form['form_submissions']),
                'content_downloads': int(request.form['content_downloads']),
                'days_since_contact': int(request.form['days_since_contact'])
            }
            
            # Train model if not already trained
            if not model.is_trained:
                flash('Training model with current data...', 'info')
                training_data = db.get_training_data()
                model.train(training_data)
                model.save_model(MODEL_PATH)
                flash('Model trained successfully!', 'success')
            
            # Get prediction
            result = model.predict_lead_score(lead_data)
            
            # Add lead to database with prediction
            lead_data.update({
                'lead_id': f'LEAD_{len(db.leads_data)+1:04d}',
                'lead_score': result['lead_score'],
                'converted': 0,  # Unknown at this point
                'created_at': datetime.now()
            })
            
            db.add_new_lead(lead_data)
            
            return render_template('score_result.html', 
                                 lead_data=lead_data, 
                                 result=result)
            
        except Exception as e:
            flash(f'Error scoring lead: {str(e)}', 'error')
            return redirect(url_for('score_lead'))
    
    return render_template('score_lead.html')

@app.route('/api/score_lead', methods=['POST'])
def api_score_lead():
    """API endpoint for scoring leads"""
    try:
        lead_data = request.json
        
        # Validate required fields
        required_fields = ['lead_source', 'industry', 'company_size', 'age', 
                          'page_views', 'time_on_site', 'email_opens', 
                          'email_clicks', 'form_submissions', 'content_downloads', 
                          'days_since_contact']
        
        for field in required_fields:
            if field not in lead_data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Train model if not already trained
        if not model.is_trained:
            training_data = db.get_training_data()
            model.train(training_data)
            model.save_model(MODEL_PATH)
        
        # Get prediction
        result = model.predict_lead_score(lead_data)
        
        return jsonify({
            'success': True,
            'lead_score': result['lead_score'],
            'conversion_probability': result['conversion_probability'],
            'predicted_conversion': result['predicted_conversion'],
            'score_category': result['score_category']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/leads')
def view_leads():
    """View all leads"""
    leads = db.leads_data.sort_values('created_at', ascending=False)
    return render_template('leads.html', leads=leads.to_dict('records'))

@app.route('/analytics')
def analytics():
    """Analytics dashboard"""
    if not model.is_trained:
        flash('Model not trained yet. Please score a lead first.', 'warning')
        return redirect(url_for('index'))
    
    # Get model metrics
    metrics = model.model_metrics
    
    # Get feature importance
    feature_importance = model.get_feature_importance()
    feature_importance_data = None
    if feature_importance is not None:
        feature_importance_data = feature_importance.head(10).to_dict('records')
    
    # Get lead distribution by score category
    leads_df = db.leads_data.copy()
    if not leads_df.empty:
        leads_df['score_category'] = leads_df['lead_score'].apply(model._categorize_score)
        score_distribution = leads_df['score_category'].value_counts().to_dict()
    else:
        score_distribution = {}
    
    return render_template('analytics.html', 
                         metrics=metrics,
                         feature_importance=feature_importance_data,
                         score_distribution=score_distribution)

@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    """Retrain the model with current data"""
    try:
        training_data = db.get_training_data()
        
        if len(training_data) < 50:
            flash('Need at least 50 leads to retrain model', 'warning')
            return redirect(url_for('analytics'))
        
        # Retrain model
        metrics = model.train(training_data)
        model.save_model(MODEL_PATH)
        
        flash(f'Model retrained successfully! New accuracy: {metrics["accuracy"]:.3f}', 'success')
        
    except Exception as e:
        flash(f'Error retraining model: {str(e)}', 'error')
    
    return redirect(url_for('analytics'))

@app.route('/export_data')
def export_data():
    """Export leads data as CSV"""
    try:
        leads_df = db.leads_data
        csv_data = leads_df.to_csv(index=False)
        
        from flask import Response
        return Response(
            csv_data,
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=leads_data.csv'}
        )
    except Exception as e:
        flash(f'Error exporting data: {str(e)}', 'error')
        return redirect(url_for('leads'))

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
