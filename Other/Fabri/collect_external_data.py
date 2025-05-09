import pandas as pd
import polars as pl
import requests
import json
from datetime import datetime
import yfinance as yf
import os

def create_data_directory():
    """Create directory structure for external data"""
    directories = [
        'external_data',
        'external_data/market_data',
        'external_data/industry_data',
        'external_data/economic_data',
        'external_data/regulatory_data'
    ]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)

def fetch_market_indices():
    """Fetch major market indices data for market context"""
    print("Fetching market indices...")
    indices = {
        '^GSPC': 'SP500',
        '^IXIC': 'NASDAQ',
        '^DJI': 'DowJones',
        '^VIX': 'Volatility'
    }
    
    for ticker, name in indices.items():
        try:
            data = yf.download(ticker, start='2015-01-01', end=datetime.now().strftime('%Y-%m-%d'))
            data.to_csv(f'external_data/market_data/{name}_data.csv')
            print(f"Successfully downloaded {name} data")
        except Exception as e:
            print(f"Error downloading {name} data: {str(e)}")

def create_industry_risk_profiles():
    """Create comprehensive industry risk classification"""
    print("Creating industry risk profiles...")
    industry_risk = {
        'technology': {
            'software': {
                'default_rate': 0.05,
                'volatility': 'high',
                'market_maturity': 'medium',
                'capital_intensity': 'low',
                'regulatory_burden': 'medium'
            },
            'hardware': {
                'default_rate': 0.06,
                'volatility': 'high',
                'market_maturity': 'high',
                'capital_intensity': 'high',
                'regulatory_burden': 'medium'
            }
        },
        'healthcare': {
            'biotech': {
                'default_rate': 0.07,
                'volatility': 'very_high',
                'market_maturity': 'low',
                'capital_intensity': 'very_high',
                'regulatory_burden': 'very_high'
            },
            'healthcare_services': {
                'default_rate': 0.03,
                'volatility': 'low',
                'market_maturity': 'high',
                'capital_intensity': 'medium',
                'regulatory_burden': 'high'
            }
        },
        'financial_services': {
            'fintech': {
                'default_rate': 0.06,
                'volatility': 'high',
                'market_maturity': 'low',
                'capital_intensity': 'medium',
                'regulatory_burden': 'high'
            },
            'traditional_banking': {
                'default_rate': 0.02,
                'volatility': 'medium',
                'market_maturity': 'very_high',
                'capital_intensity': 'high',
                'regulatory_burden': 'very_high'
            }
        },
        'retail': {
            'e_commerce': {
                'default_rate': 0.08,
                'volatility': 'high',
                'market_maturity': 'medium',
                'capital_intensity': 'medium',
                'regulatory_burden': 'low'
            },
            'traditional_retail': {
                'default_rate': 0.04,
                'volatility': 'medium',
                'market_maturity': 'high',
                'capital_intensity': 'high',
                'regulatory_burden': 'medium'
            }
        }
    }
    
    with open('external_data/industry_data/industry_risk_profiles.json', 'w') as f:
        json.dump(industry_risk, f, indent=4)
    print("Industry risk profiles created")

def create_regulatory_framework():
    """Create regulatory environment scoring system"""
    print("Creating regulatory framework data...")
    regulatory_framework = {
        'USA': {
            'regulatory_quality': 0.89,
            'rule_of_law': 0.88,
            'control_of_corruption': 0.85,
            'startup_regulation_index': 0.92,
            'investor_protection': 0.90,
            'intellectual_property_rights': 0.88,
            'financial_regulation': {
                'banking_supervision': 0.90,
                'securities_regulation': 0.92,
                'insurance_regulation': 0.88
            }
        },
        'GBR': {
            'regulatory_quality': 0.87,
            'rule_of_law': 0.85,
            'control_of_corruption': 0.87,
            'startup_regulation_index': 0.89,
            'investor_protection': 0.88,
            'intellectual_property_rights': 0.86,
            'financial_regulation': {
                'banking_supervision': 0.88,
                'securities_regulation': 0.90,
                'insurance_regulation': 0.87
            }
        },
        'DEU': {
            'regulatory_quality': 0.88,
            'rule_of_law': 0.89,
            'control_of_corruption': 0.86,
            'startup_regulation_index': 0.85,
            'investor_protection': 0.85,
            'intellectual_property_rights': 0.87,
            'financial_regulation': {
                'banking_supervision': 0.89,
                'securities_regulation': 0.88,
                'insurance_regulation': 0.89
            }
        }
    }
    
    with open('external_data/regulatory_data/regulatory_framework.json', 'w') as f:
        json.dump(regulatory_framework, f, indent=4)
    print("Regulatory framework data created")

def create_economic_indicators():
    """Create economic indicators database"""
    print("Creating economic indicators database...")
    try:
        # Read World Bank GDP data
        wb_gdp_data = pd.read_csv('Data/World Bank GDP data/API_NY.GDP.PCAP.KD_DS2_en_csv_v2_13354.csv', skiprows=4)
        wb_gdp_data.to_csv('external_data/economic_data/world_bank_gdp.csv', index=False)
        print("World Bank GDP data processed")
        
        # Process IMF data using openpyxl engine
        gdp_data = pd.read_excel('Romain/imf-dm-export-20250330_gdp_capita_ppi.xls', engine='openpyxl')
        unemployment_data = pd.read_excel('Romain/imf-dm-export-20250330_unempl.xls', engine='openpyxl')
        inflation_data = pd.read_excel('Romain/imf-dm-export-20250330_infl.xls', engine='openpyxl')
        
        # Save processed versions
        gdp_data.to_csv('external_data/economic_data/imf_gdp_per_capita.csv', index=False)
        unemployment_data.to_csv('external_data/economic_data/imf_unemployment.csv', index=False)
        inflation_data.to_csv('external_data/economic_data/imf_inflation.csv', index=False)
        print("IMF economic indicators processed and saved")
    except Exception as e:
        print(f"Error processing economic data: {str(e)}")

def main():
    # Create directory structure
    create_data_directory()
    
    # Fetch market data
    fetch_market_indices()
    
    # Create industry risk profiles
    create_industry_risk_profiles()
    
    # Create regulatory framework
    create_regulatory_framework()
    
    # Process economic indicators
    create_economic_indicators()
    
    print("\nData collection and processing complete!")

if __name__ == "__main__":
    main() 