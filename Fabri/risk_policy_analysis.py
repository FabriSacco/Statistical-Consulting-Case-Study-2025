import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from credit_risk_assessment import CreditRiskAssessment, CreditRiskPolicy, RiskCategory
from typing import Dict, List, Any
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class RiskPolicyAnalysis:
    def __init__(self):
        self.policy = CreditRiskPolicy()
        self.risk_assessor = CreditRiskAssessment(self.policy)
        
        # Set style for visualizations
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def generate_sample_portfolio(self, n_startups: int = 100) -> List[Dict[str, Any]]:
        """Generate a sample portfolio of startups for analysis"""
        industries = [
            ('technology', ['software', 'hardware', 'fintech', 'biotech']),
            ('healthcare', ['medical_devices', 'pharmaceuticals', 'healthcare_it']),
            ('finance', ['banking', 'insurance', 'investment']),
            ('retail', ['e-commerce', 'brick_and_mortar', 'omnichannel'])
        ]
        
        countries = ['USA', 'UK', 'DEU', 'FRA', 'JPN', 'CHN', 'IND']
        current_year = 2024
        
        portfolio = []
        for _ in range(n_startups):
            industry, sub_industries = industries[np.random.randint(0, len(industries))]
            portfolio.append({
                'name': f'Startup_{_}',
                'industry': industry,
                'sub_industry': sub_industries[np.random.randint(0, len(sub_industries))],
                'country_code': countries[np.random.randint(0, len(countries))],
                'foundation_year': np.random.randint(current_year - 5, current_year + 1),
                'funding_total_usd': np.random.randint(100000, 10000000),
                'funding_rounds': np.random.randint(1, 5),
                'loan_amount': np.random.randint(50000, 2000000)
            })
        
        return portfolio
    
    def analyze_portfolio(self, portfolio: List[Dict[str, Any]]) -> pd.DataFrame:
        """Analyze a portfolio of startups and return results as a DataFrame"""
        results = []
        for startup in portfolio:
            loan_amount = startup.pop('loan_amount')  # Remove loan_amount from startup dict
            assessment = self.risk_assessor.assess_credit_risk(startup, loan_amount)
            
            results.append({
                **startup,
                'loan_amount': loan_amount,
                'risk_score': assessment['total_risk_score'],
                'risk_category': assessment['risk_category'],
                'market_risk': assessment['component_scores']['market_risk'],
                'industry_risk': assessment['component_scores']['industry_risk'],
                'regulatory_risk': assessment['component_scores']['regulatory_risk'],
                'economic_risk': assessment['component_scores']['economic_risk'],
                'interest_rate': assessment['loan_terms']['interest_rate']
            })
        
        return pd.DataFrame(results)
    
    def plot_risk_distribution(self, df: pd.DataFrame, save_path: str = 'visualizations'):
        """Plot the distribution of risk scores and categories"""
        # Create visualizations directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Risk Score Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='risk_score', bins=30)
        plt.title('Distribution of Risk Scores')
        plt.xlabel('Risk Score')
        plt.ylabel('Count')
        plt.savefig(f'{save_path}/risk_score_distribution.png')
        plt.close()
        
        # Risk Categories
        plt.figure(figsize=(10, 6))
        risk_counts = df['risk_category'].value_counts()
        sns.barplot(x=risk_counts.index, y=risk_counts.values)
        plt.title('Distribution of Risk Categories')
        plt.xlabel('Risk Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_path}/risk_category_distribution.png')
        plt.close()
    
    def plot_risk_components(self, df: pd.DataFrame, save_path: str = 'visualizations'):
        """Plot analysis of risk components"""
        # Risk Components Box Plot
        plt.figure(figsize=(12, 6))
        risk_components = df[['market_risk', 'industry_risk', 'regulatory_risk', 'economic_risk']]
        sns.boxplot(data=risk_components)
        plt.title('Distribution of Risk Components')
        plt.ylabel('Risk Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_path}/risk_components_distribution.png')
        plt.close()
        
        # Risk Components by Industry
        plt.figure(figsize=(15, 8))
        industry_risks = df.groupby('industry')[['market_risk', 'industry_risk', 'regulatory_risk', 'economic_risk']].mean()
        industry_risks.plot(kind='bar')
        plt.title('Average Risk Components by Industry')
        plt.xlabel('Industry')
        plt.ylabel('Risk Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{save_path}/industry_risk_components.png')
        plt.close()
    
    def plot_portfolio_metrics(self, df: pd.DataFrame, save_path: str = 'visualizations'):
        """Plot portfolio-level metrics"""
        # Interest Rate vs Risk Score
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='risk_score', y='interest_rate', hue='industry')
        plt.title('Interest Rate vs Risk Score')
        plt.xlabel('Risk Score')
        plt.ylabel('Interest Rate')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{save_path}/interest_rate_vs_risk.png')
        plt.close()
        
        # Loan Amount Distribution by Risk Category
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='risk_category', y='loan_amount')
        plt.title('Loan Amount Distribution by Risk Category')
        plt.xlabel('Risk Category')
        plt.ylabel('Loan Amount (USD)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_path}/loan_amount_by_risk.png')
        plt.close()
    
    def generate_policy_recommendations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate policy recommendations based on portfolio analysis"""
        recommendations = {
            'portfolio_summary': {
                'total_startups': len(df),
                'total_loan_amount': df['loan_amount'].sum(),
                'average_risk_score': df['risk_score'].mean(),
                'risk_category_distribution': df['risk_category'].value_counts().to_dict()
            },
            'risk_insights': {
                'highest_risk_industries': df.groupby('industry')['risk_score'].mean().sort_values(ascending=False).head(3).to_dict(),
                'lowest_risk_industries': df.groupby('industry')['risk_score'].mean().sort_values().head(3).to_dict(),
                'country_risk_levels': df.groupby('country_code')['risk_score'].mean().to_dict()
            },
            'policy_suggestions': [
                {
                    'area': 'Risk Thresholds',
                    'current': {
                        'low': self.policy.low_risk_threshold,
                        'medium': self.policy.medium_risk_threshold,
                        'high': self.policy.high_risk_threshold
                    },
                    'suggested': {
                        'low': np.percentile(df['risk_score'], 25),
                        'medium': np.percentile(df['risk_score'], 50),
                        'high': np.percentile(df['risk_score'], 75)
                    }
                },
                {
                    'area': 'Industry Exposure',
                    'current': self.policy.max_exposure_per_industry,
                    'suggested': min(0.20, df.groupby('industry')['loan_amount'].sum().max() / df['loan_amount'].sum())
                },
                {
                    'area': 'Interest Rate Structure',
                    'current': {
                        'base_rate': self.policy.base_interest_rate,
                        'max_premium': max(self.policy.risk_premiums.values())
                    },
                    'suggested': {
                        'base_rate': max(0.04, np.percentile(df['interest_rate'], 25)),
                        'max_premium': min(0.15, np.percentile(df['interest_rate'], 95) - self.policy.base_interest_rate)
                    }
                }
            ]
        }
        
        return recommendations

def main():
    # Initialize analysis
    analyzer = RiskPolicyAnalysis()
    
    # Generate and analyze sample portfolio
    portfolio = analyzer.generate_sample_portfolio(n_startups=100)
    results_df = analyzer.analyze_portfolio(portfolio)
    
    # Generate visualizations
    analyzer.plot_risk_distribution(results_df)
    analyzer.plot_risk_components(results_df)
    analyzer.plot_portfolio_metrics(results_df)
    
    # Generate policy recommendations
    recommendations = analyzer.generate_policy_recommendations(results_df)
    
    # Save recommendations to file
    with open('visualizations/policy_recommendations.json', 'w') as f:
        json.dump(recommendations, f, indent=4, cls=NumpyEncoder)
    
    print("Analysis complete. Visualizations and recommendations have been saved to the 'visualizations' directory.")

if __name__ == "__main__":
    main() 