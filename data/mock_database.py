"""
Mock database module for lead scoring model
Generates realistic lead data for training and testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class MockLeadDatabase:
    def __init__(self):
        self.leads_data = None
        self._generate_mock_data()
    
    def _generate_mock_data(self, n_samples=1000):
        """Generate realistic lead data"""
        np.random.seed(42)
        random.seed(42)
        
        # Lead sources with different conversion probabilities
        lead_sources = ['Email', 'Website', 'Social Media', 'Referral', 'Paid Ad', 'Webinar', 'Trade Show', 'Cold Call']
        source_conversion_rates = [0.15, 0.08, 0.12, 0.35, 0.18, 0.45, 0.25, 0.05]
        
        # Industries with different conversion rates
        industries = ['Technology', 'Healthcare', 'Finance', 'Education', 'Retail', 'Manufacturing', 'Real Estate', 'Consulting']
        industry_conversion_rates = [0.25, 0.20, 0.30, 0.15, 0.12, 0.18, 0.22, 0.28]
        
        # Company sizes
        company_sizes = ['Startup (1-10)', 'Small (11-50)', 'Medium (51-200)', 'Large (201-1000)', 'Enterprise (1000+)']
        size_conversion_rates = [0.10, 0.15, 0.25, 0.30, 0.35]
        
        data = []
        
        for i in range(n_samples):
            # Basic demographics
            age = np.random.normal(35, 10)
            age = max(22, min(65, int(age)))  # Clamp between 22-65
            
            # Lead source (affects conversion probability)
            lead_source = np.random.choice(lead_sources, p=[0.2, 0.25, 0.15, 0.1, 0.15, 0.05, 0.05, 0.05])
            source_idx = lead_sources.index(lead_source)
            base_conversion_prob = source_conversion_rates[source_idx]
            
            # Industry
            industry = np.random.choice(industries, p=[0.15, 0.12, 0.13, 0.10, 0.15, 0.12, 0.08, 0.15])
            industry_idx = industries.index(industry)
            industry_factor = industry_conversion_rates[industry_idx] / 0.2  # Normalize around 0.2
            
            # Company size
            company_size = np.random.choice(company_sizes, p=[0.15, 0.25, 0.30, 0.20, 0.10])
            size_idx = company_sizes.index(company_size)
            size_factor = size_conversion_rates[size_idx] / 0.23  # Normalize around 0.23
            
            # Behavioral data (correlated with conversion)
            page_views = max(1, int(np.random.exponential(8)))
            time_on_site = max(30, int(np.random.exponential(600)))  # seconds
            
            # Email engagement
            email_opens = max(0, int(np.random.poisson(3)))
            email_clicks = max(0, int(np.random.poisson(1.5)))
            
            # Form submissions and downloads
            form_submissions = max(0, int(np.random.poisson(0.8)))
            content_downloads = max(0, int(np.random.poisson(1.2)))
            
            # Days since first contact
            days_since_contact = max(1, int(np.random.exponential(15)))
            
            # Calculate conversion probability based on features
            behavioral_score = (
                (page_views / 10) * 0.2 +
                (time_on_site / 1000) * 0.15 +
                (email_opens / 5) * 0.1 +
                (email_clicks / 3) * 0.15 +
                (form_submissions / 2) * 0.2 +
                (content_downloads / 3) * 0.1 +
                (1 / (days_since_contact / 10 + 1)) * 0.1
            )
            
            # Final conversion probability
            conversion_prob = base_conversion_prob * industry_factor * size_factor * (1 + behavioral_score)
            conversion_prob = min(0.9, max(0.01, conversion_prob))  # Clamp between 1% and 90%
            
            # Determine conversion
            converted = 1 if np.random.random() < conversion_prob else 0
            
            # Lead score (what we want to predict)
            lead_score = int(conversion_prob * 100)
            
            data.append({
                'lead_id': f'LEAD_{i+1:04d}',
                'lead_source': lead_source,
                'industry': industry,
                'company_size': company_size,
                'age': age,
                'page_views': page_views,
                'time_on_site': time_on_site,
                'email_opens': email_opens,
                'email_clicks': email_clicks,
                'form_submissions': form_submissions,
                'content_downloads': content_downloads,
                'days_since_contact': days_since_contact,
                'lead_score': lead_score,
                'converted': converted,
                'created_at': datetime.now() - timedelta(days=days_since_contact)
            })
        
        self.leads_data = pd.DataFrame(data)
        return self.leads_data
    
    def get_training_data(self):
        """Get data for model training"""
        return self.leads_data.copy()
    
    def get_lead_by_id(self, lead_id):
        """Get specific lead by ID"""
        return self.leads_data[self.leads_data['lead_id'] == lead_id].iloc[0] if not self.leads_data[self.leads_data['lead_id'] == lead_id].empty else None
    
    def add_new_lead(self, lead_data):
        """Add a new lead to the database"""
        new_lead = pd.DataFrame([lead_data])
        self.leads_data = pd.concat([self.leads_data, new_lead], ignore_index=True)
        return lead_data['lead_id']
    
    def get_conversion_stats(self):
        """Get conversion statistics"""
        total_leads = len(self.leads_data)
        converted_leads = self.leads_data['converted'].sum()
        conversion_rate = converted_leads / total_leads if total_leads > 0 else 0
        
        return {
            'total_leads': total_leads,
            'converted_leads': converted_leads,
            'conversion_rate': conversion_rate,
            'avg_lead_score': self.leads_data['lead_score'].mean()
        }

if __name__ == "__main__":
    # Test the mock database
    db = MockLeadDatabase()
    print("Mock database created successfully!")
    print(f"Generated {len(db.leads_data)} leads")
    print("\nSample data:")
    print(db.leads_data.head())
    print("\nConversion stats:")
    print(db.get_conversion_stats())
