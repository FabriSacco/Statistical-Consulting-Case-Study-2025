import pandas as pd
import polars as pl
import json
import numpy as np
from datetime import datetime
import os
from typing import Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

class RiskCategory(Enum):
    LOW = "Low Risk"
    MEDIUM = "Medium Risk"
    HIGH = "High Risk"
    VERY_HIGH = "Very High Risk"

def default_risk_premiums():
    return {
        RiskCategory.LOW: 0.02,        # +2% for low risk
        RiskCategory.MEDIUM: 0.04,     # +4% for medium risk
        RiskCategory.HIGH: 0.07,       # +7% for high risk
        RiskCategory.VERY_HIGH: 0.10   # +10% for very high risk
    }

@dataclass
class CreditRiskPolicy:
    """Defines the bank's credit risk policy parameters"""
    # Risk thresholds
    low_risk_threshold: float = 0.3
    medium_risk_threshold: float = 0.6
    high_risk_threshold: float = 0.8
    
    # Component weights
    market_risk_weight: float = 0.25
    industry_risk_weight: float = 0.30
    regulatory_risk_weight: float = 0.20
    economic_risk_weight: float = 0.25
    
    # Policy limits
    max_exposure_per_industry: float = 0.15  # Maximum 15% exposure per industry
    max_exposure_per_country: float = 0.25   # Maximum 25% exposure per country
    min_collateral_ratio: float = 0.3        # Minimum 30% collateral coverage
    max_loan_term_years: int = 5             # Maximum loan term
    
    # Monitoring requirements
    low_risk_monitoring: str = "Quarterly"
    medium_risk_monitoring: str = "Monthly"
    high_risk_monitoring: str = "Weekly"
    very_high_risk_monitoring: str = "Daily"
    
    # Interest rate adjustments
    base_interest_rate: float = 0.05  # 5% base rate
    risk_premiums: Dict[RiskCategory, float] = field(default_factory=default_risk_premiums)

class CreditRiskAssessment:
    def __init__(self, policy: CreditRiskPolicy = None):
        self.policy = policy or CreditRiskPolicy()
        
        # Load all data sources
        self.load_data()
        
        # Initialize exposure tracking
        self.industry_exposure = {}
        self.country_exposure = {}
        
    def load_data(self):
        """Load all required data sources"""
        # Market data
        self.market_data = {}
        for index in ['SP500', 'NASDAQ', 'DowJones', 'Volatility']:
            try:
                df = pd.read_csv(f'external_data/market_data/{index}_data.csv')
                # Convert numeric columns to float
                numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                self.market_data[index] = df
            except FileNotFoundError:
                print(f"Warning: {index} data not found")
        
        # Industry risk profiles
        with open('external_data/industry_data/industry_risk_profiles.json', 'r') as f:
            self.industry_risk = json.load(f)
        
        # Regulatory framework
        with open('external_data/regulatory_data/regulatory_framework.json', 'r') as f:
            self.regulatory_framework = json.load(f)
        
        # Economic indicators
        try:
            self.gdp_data = pd.read_csv('external_data/economic_data/world_bank_gdp.csv')
        except FileNotFoundError:
            print("Warning: GDP data not found")
    
    def check_policy_compliance(self, startup_data: Dict[str, Any], loan_amount: float) -> Dict[str, Any]:
        """Check if the proposed loan complies with bank policy"""
        industry = startup_data.get('industry', '')
        country = startup_data.get('country_code', '')
        
        # Update exposure tracking
        self.industry_exposure[industry] = self.industry_exposure.get(industry, 0) + loan_amount
        self.country_exposure[country] = self.country_exposure.get(country, 0) + loan_amount
        
        # Calculate total exposure
        total_exposure = sum(self.industry_exposure.values())
        
        compliance_checks = {
            'industry_exposure': {
                'compliant': self.industry_exposure[industry] / total_exposure <= self.policy.max_exposure_per_industry,
                'current_ratio': self.industry_exposure[industry] / total_exposure,
                'max_allowed': self.policy.max_exposure_per_industry
            },
            'country_exposure': {
                'compliant': self.country_exposure[country] / total_exposure <= self.policy.max_exposure_per_country,
                'current_ratio': self.country_exposure[country] / total_exposure,
                'max_allowed': self.policy.max_exposure_per_country
            }
        }
        
        return compliance_checks

    def calculate_market_risk(self, startup_data: Dict[str, Any]) -> float:
        """Calculate market risk score based on market conditions"""
        # Get the most recent market data
        latest_market_data = {}
        for index, df in self.market_data.items():
            if not df.empty:
                latest_market_data[index] = df.iloc[-1]
        
        # Calculate market volatility score
        volatility_score = 0.0
        if 'Volatility' in latest_market_data:
            vix = float(latest_market_data['Volatility']['Close'])
            if vix < 15:
                volatility_score = 0.2
            elif vix < 25:
                volatility_score = 0.5
            else:
                volatility_score = 0.8
        
        # Calculate market trend score
        trend_score = 0.0
        for index in ['SP500', 'NASDAQ', 'DowJones']:
            if index in latest_market_data:
                df = self.market_data[index]
                if len(df) >= 200:  # Check for enough data points
                    ma200 = float(df['Close'].rolling(window=200).mean().iloc[-1])
                    current_price = float(df['Close'].iloc[-1])
                    if current_price > ma200:
                        trend_score += 0.2
                    else:
                        trend_score += 0.8
        
        # Average the trend scores
        trend_score = trend_score / len([i for i in ['SP500', 'NASDAQ', 'DowJones'] if i in latest_market_data])
        
        # Combine scores
        market_risk = (volatility_score * 0.4) + (trend_score * 0.6)
        return market_risk
    
    def calculate_industry_risk(self, startup_data: Dict[str, Any]) -> float:
        """Calculate industry-specific risk score"""
        industry = startup_data.get('industry', '')
        sub_industry = startup_data.get('sub_industry', '')
        
        # Default to highest risk if industry not found
        if industry not in self.industry_risk:
            return 0.8
        
        if sub_industry not in self.industry_risk[industry]:
            return 0.8
        
        industry_profile = self.industry_risk[industry][sub_industry]
        
        # Convert qualitative measures to numerical scores
        volatility_map = {
            'very_high': 0.9,
            'high': 0.7,
            'medium': 0.5,
            'low': 0.3
        }
        
        maturity_map = {
            'very_high': 0.2,
            'high': 0.4,
            'medium': 0.6,
            'low': 0.8
        }
        
        # Calculate weighted industry risk
        industry_risk = (
            industry_profile['default_rate'] * 0.4 +
            volatility_map[industry_profile['volatility']] * 0.3 +
            maturity_map[industry_profile['market_maturity']] * 0.3
        )
        
        return industry_risk
    
    def calculate_regulatory_risk(self, startup_data: Dict[str, Any]) -> float:
        """Calculate regulatory risk score based on country and industry"""
        country = startup_data.get('country_code', '')
        industry = startup_data.get('industry', '')
        
        if country not in self.regulatory_framework:
            return 0.8  # Default to high risk for unknown countries
        
        regulatory_data = self.regulatory_framework[country]
        
        # Calculate base regulatory risk
        regulatory_risk = (
            (1 - regulatory_data['regulatory_quality']) * 0.3 +
            (1 - regulatory_data['rule_of_law']) * 0.3 +
            (1 - regulatory_data['control_of_corruption']) * 0.4
        )
        
        # Adjust for industry-specific regulatory burden
        if industry in self.industry_risk:
            industry_profile = next(iter(self.industry_risk[industry].values()))
            regulatory_burden_map = {
                'very_high': 0.8,
                'high': 0.6,
                'medium': 0.4,
                'low': 0.2
            }
            regulatory_burden = regulatory_burden_map[industry_profile['regulatory_burden']]
            regulatory_risk = (regulatory_risk * 0.7) + (regulatory_burden * 0.3)
        
        return regulatory_risk
    
    def calculate_economic_risk(self, startup_data: Dict[str, Any]) -> float:
        """Calculate economic risk score based on country and timing"""
        country = startup_data.get('country_code', '')
        founded_year = startup_data.get('foundation_year', datetime.now().year)
        
        if not hasattr(self, 'gdp_data') or self.gdp_data.empty:
            return 0.5  # Default to medium risk if no GDP data
        
        # Get GDP data for the country
        country_gdp = self.gdp_data[self.gdp_data['Country Code'] == country]
        if country_gdp.empty:
            return 0.5
        
        # Calculate GDP growth rate
        gdp_columns = [col for col in country_gdp.columns if col.isdigit()]
        if len(gdp_columns) < 2:
            return 0.5
        
        recent_gdp = float(country_gdp[gdp_columns[-1]].values[0])
        previous_gdp = float(country_gdp[gdp_columns[-2]].values[0])
        
        if previous_gdp == 0:
            return 0.5
        
        gdp_growth = (recent_gdp - previous_gdp) / previous_gdp
        
        # Convert GDP growth to risk score
        if gdp_growth > 0.05:
            economic_risk = 0.2
        elif gdp_growth > 0:
            economic_risk = 0.4
        elif gdp_growth > -0.02:
            economic_risk = 0.6
        else:
            economic_risk = 0.8
        
        return economic_risk
    
    def assess_credit_risk(self, startup_data: Dict[str, Any], loan_amount: float) -> Dict[str, Any]:
        """Main method to assess credit risk for a startup"""
        # Check policy compliance
        compliance_checks = self.check_policy_compliance(startup_data, loan_amount)
        
        # Calculate individual risk components
        market_risk = self.calculate_market_risk(startup_data)
        industry_risk = self.calculate_industry_risk(startup_data)
        regulatory_risk = self.calculate_regulatory_risk(startup_data)
        economic_risk = self.calculate_economic_risk(startup_data)
        
        # Calculate weighted total risk score
        total_risk = (
            market_risk * self.policy.market_risk_weight +
            industry_risk * self.policy.industry_risk_weight +
            regulatory_risk * self.policy.regulatory_risk_weight +
            economic_risk * self.policy.economic_risk_weight
        )
        
        # Determine risk category
        if total_risk <= self.policy.low_risk_threshold:
            risk_category = RiskCategory.LOW
        elif total_risk <= self.policy.medium_risk_threshold:
            risk_category = RiskCategory.MEDIUM
        elif total_risk <= self.policy.high_risk_threshold:
            risk_category = RiskCategory.HIGH
        else:
            risk_category = RiskCategory.VERY_HIGH
        
        # Calculate interest rate based on risk category
        interest_rate = self.policy.base_interest_rate + self.policy.risk_premiums[risk_category]
        
        # Prepare detailed risk report
        risk_report = {
            'total_risk_score': total_risk,
            'risk_category': risk_category.value,
            'component_scores': {
                'market_risk': market_risk,
                'industry_risk': industry_risk,
                'regulatory_risk': regulatory_risk,
                'economic_risk': economic_risk
            },
            'policy_compliance': compliance_checks,
            'loan_terms': {
                'interest_rate': interest_rate,
                'max_term_years': self.policy.max_loan_term_years,
                'min_collateral_ratio': self.policy.min_collateral_ratio,
                'monitoring_frequency': getattr(self.policy, f"{risk_category.name.lower()}_risk_monitoring")
            },
            'recommendations': self.generate_recommendations(total_risk, risk_category)
        }
        
        return risk_report
    
    def generate_recommendations(self, total_risk: float, risk_category: RiskCategory) -> List[str]:
        """Generate recommendations based on risk assessment"""
        recommendations = []
        
        if risk_category == RiskCategory.LOW:
            recommendations.extend([
                "Consider standard lending terms",
                "Regular monitoring (quarterly)",
                "Standard collateral requirements"
            ])
        elif risk_category == RiskCategory.MEDIUM:
            recommendations.extend([
                "Consider higher interest rates",
                "More frequent monitoring (monthly)",
                "Increased collateral requirements",
                "Consider shorter loan terms"
            ])
        elif risk_category == RiskCategory.HIGH:
            recommendations.extend([
                "Require significant collateral",
                "Weekly monitoring",
                "Higher interest rates",
                "Shorter loan terms",
                "Consider requiring personal guarantees"
            ])
        else:  # RiskCategory.VERY_HIGH
            recommendations.extend([
                "Consider declining the application",
                "If proceeding, require maximum collateral",
                "Daily monitoring",
                "Highest interest rates",
                "Shortest possible loan terms",
                "Require personal guarantees",
                "Consider requiring additional investors"
            ])
        
        return recommendations

def main():
    # Initialize credit risk policy
    policy = CreditRiskPolicy()
    
    # Example usage
    risk_assessor = CreditRiskAssessment(policy)
    
    # Example startup data
    example_startup = {
        'name': 'Example Tech Startup',
        'industry': 'technology',
        'sub_industry': 'software',
        'country_code': 'USA',
        'foundation_year': 2020,
        'funding_total_usd': 1000000,
        'funding_rounds': 2
    }
    
    # Example loan amount
    loan_amount = 500000
    
    # Assess credit risk
    risk_report = risk_assessor.assess_credit_risk(example_startup, loan_amount)
    
    # Print results
    print("\nCredit Risk Assessment Report")
    print("=" * 50)
    print(f"Total Risk Score: {risk_report['total_risk_score']:.2f}")
    print(f"Risk Category: {risk_report['risk_category']}")
    print("\nComponent Scores:")
    for component, score in risk_report['component_scores'].items():
        print(f"- {component.replace('_', ' ').title()}: {score:.2f}")
    
    print("\nPolicy Compliance:")
    for check, details in risk_report['policy_compliance'].items():
        status = "Compliant" if details['compliant'] else "Not Compliant"
        print(f"- {check.replace('_', ' ').title()}: {status}")
        print(f"  Current Ratio: {details['current_ratio']:.2%}")
        print(f"  Maximum Allowed: {details['max_allowed']:.2%}")
    
    print("\nProposed Loan Terms:")
    for term, value in risk_report['loan_terms'].items():
        if isinstance(value, float):
            print(f"- {term.replace('_', ' ').title()}: {value:.2%}")
        else:
            print(f"- {term.replace('_', ' ').title()}: {value}")
    
    print("\nRecommendations:")
    for rec in risk_report['recommendations']:
        print(f"- {rec}")

if __name__ == "__main__":
    main() 