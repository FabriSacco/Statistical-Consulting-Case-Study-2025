import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from risk_policy_analysis import RiskPolicyAnalysis
import json
from typing import Dict, Any
from datetime import datetime
import os

class CreditRiskDashboard:
    def __init__(self):
        self.analyzer = RiskPolicyAnalysis()
        self.portfolio = self.analyzer.generate_sample_portfolio(n_startups=100)
        self.results_df = self.analyzer.analyze_portfolio(self.portfolio)
        
        # Set page config
        st.set_page_config(
            page_title="Credit Risk Assessment Dashboard",
            page_icon="📊",
            layout="wide"
        )
    
    def create_sidebar(self):
        """Create sidebar filters and controls"""
        st.sidebar.title("Filters")
        
        # Industry filter
        industries = self.results_df['industry'].unique()
        selected_industries = st.sidebar.multiselect(
            "Select Industries",
            options=industries,
            default=industries
        )
        
        # Risk category filter
        risk_categories = self.results_df['risk_category'].unique()
        selected_risk_categories = st.sidebar.multiselect(
            "Select Risk Categories",
            options=risk_categories,
            default=risk_categories
        )
        
        # Country filter
        countries = self.results_df['country_code'].unique()
        selected_countries = st.sidebar.multiselect(
            "Select Countries",
            options=countries,
            default=countries
        )
        
        # Risk score range
        min_risk, max_risk = st.sidebar.slider(
            "Risk Score Range",
            min_value=float(self.results_df['risk_score'].min()),
            max_value=float(self.results_df['risk_score'].max()),
            value=(float(self.results_df['risk_score'].min()), float(self.results_df['risk_score'].max()))
        )
        
        # Apply filters
        filtered_df = self.results_df[
            (self.results_df['industry'].isin(selected_industries)) &
            (self.results_df['risk_category'].isin(selected_risk_categories)) &
            (self.results_df['country_code'].isin(selected_countries)) &
            (self.results_df['risk_score'] >= min_risk) &
            (self.results_df['risk_score'] <= max_risk)
        ]
        
        return filtered_df
    
    def create_summary_metrics(self, df: pd.DataFrame):
        """Create summary metrics cards"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Portfolio",
                f"${df['loan_amount'].sum():,.0f}",
                f"{len(df)} startups"
            )
        
        with col2:
            avg_risk = df['risk_score'].mean()
            st.metric(
                "Average Risk Score",
                f"{avg_risk:.2f}",
                "Lower is better"
            )
        
        with col3:
            avg_interest = df['interest_rate'].mean()
            st.metric(
                "Average Interest Rate",
                f"{avg_interest:.1%}",
                "Risk-adjusted"
            )
        
        with col4:
            risk_diversity = len(df['risk_category'].unique()) / len(df['risk_category'].unique())
            st.metric(
                "Risk Diversity",
                f"{risk_diversity:.0%}",
                "Portfolio coverage"
            )
    
    def create_risk_distribution_plot(self, df: pd.DataFrame):
        """Create risk distribution visualization"""
        st.subheader("Risk Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df,
                x='risk_score',
                nbins=30,
                title='Risk Score Distribution',
                labels={'risk_score': 'Risk Score', 'count': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                df,
                names='risk_category',
                title='Risk Category Distribution',
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def create_risk_components_plot(self, df: pd.DataFrame):
        """Create risk components visualization"""
        st.subheader("Risk Components Analysis")
        
        # Calculate average risk components
        risk_components = df[['market_risk', 'industry_risk', 'regulatory_risk', 'economic_risk']].mean()
        
        fig = go.Figure(data=[
            go.Bar(
                x=risk_components.index,
                y=risk_components.values,
                text=risk_components.values.round(2),
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Average Risk Components',
            xaxis_title='Risk Component',
            yaxis_title='Risk Score',
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_industry_analysis(self, df: pd.DataFrame):
        """Create industry-specific analysis"""
        st.subheader("Industry Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            industry_risks = df.groupby('industry')[['market_risk', 'industry_risk', 'regulatory_risk', 'economic_risk']].mean()
            fig = px.bar(
                industry_risks,
                title='Risk Components by Industry',
                labels={'value': 'Risk Score', 'variable': 'Risk Component'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            industry_metrics = df.groupby('industry').agg({
                'risk_score': 'mean',
                'interest_rate': 'mean',
                'loan_amount': 'sum'
            }).reset_index()
            
            fig = px.scatter(
                industry_metrics,
                x='risk_score',
                y='interest_rate',
                size='loan_amount',
                text='industry',
                title='Industry Risk-Return Profile'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def create_portfolio_diversification(self, df: pd.DataFrame):
        """Create portfolio diversification analysis"""
        st.subheader("Portfolio Diversification")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                df,
                names='industry',
                values='loan_amount',
                title='Portfolio Distribution by Industry'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                df,
                names='country_code',
                values='loan_amount',
                title='Portfolio Distribution by Country'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def create_risk_return_tradeoff(self, df: pd.DataFrame):
        """Create risk-return tradeoff visualization"""
        st.subheader("Risk-Return Analysis")
        
        fig = px.scatter(
            df,
            x='risk_score',
            y='interest_rate',
            color='industry',
            size='loan_amount',
            hover_data=['name', 'country_code', 'foundation_year'],
            title='Risk-Return Tradeoff by Industry'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Run the dashboard"""
        st.title("Credit Risk Assessment Dashboard")
        
        # Title and description
        st.markdown("""
        This interactive dashboard provides insights into credit risk assessment and portfolio analysis.
        Use the filters and controls to explore different aspects of the credit risk data.
        """)
        
        # Create sidebar and get filtered data
        filtered_df = self.create_sidebar()
        
        # Create summary metrics
        self.create_summary_metrics(filtered_df)
        
        # Create visualizations
        self.create_risk_distribution_plot(filtered_df)
        self.create_risk_components_plot(filtered_df)
        self.create_industry_analysis(filtered_df)
        self.create_portfolio_diversification(filtered_df)
        self.create_risk_return_tradeoff(filtered_df)
        
        # Add download button for filtered data
        st.download_button(
            label="Download Filtered Data",
            data=filtered_df.to_csv(index=False),
            file_name="filtered_portfolio_data.csv",
            mime="text/csv"
        )

def main():
    dashboard = CreditRiskDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()

# Load data
@st.cache_data
def load_data():
    try:
        # Load market data
        market_data = {}
        market_files = ['SP500_data.csv', 'NASDAQ_data.csv', 'DowJones_data.csv', 'Volatility_data.csv']
        for file in market_files:
            if os.path.exists(f'external_data/market_data/{file}'):
                market_data[file.split('_')[0]] = pd.read_csv(f'external_data/market_data/{file}')
        
        # Load industry risk profiles
        industry_risk = {}
        if os.path.exists('external_data/industry_data/industry_risk_profiles.json'):
            with open('external_data/industry_data/industry_risk_profiles.json', 'r') as f:
                industry_risk = json.load(f)
        
        # Load regulatory framework
        regulatory_data = {}
        if os.path.exists('external_data/regulatory_data/regulatory_framework.json'):
            with open('external_data/regulatory_data/regulatory_framework.json', 'r') as f:
                regulatory_data = json.load(f)
        
        # Load economic data
        economic_data = {}
        if os.path.exists('external_data/economic_data/world_bank_gdp.csv'):
            economic_data['gdp'] = pd.read_csv('external_data/economic_data/world_bank_gdp.csv')
        
        return {
            'market_data': market_data,
            'industry_risk': industry_risk,
            'regulatory_data': regulatory_data,
            'economic_data': economic_data
        }
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load the data
data = load_data()

if data is not None:
    # Market Data Analysis
    st.header("Market Data Analysis")
    
    # Create tabs for different market indices
    tab1, tab2, tab3, tab4 = st.tabs(["S&P 500", "NASDAQ", "Dow Jones", "Volatility"])
    
    with tab1:
        if 'SP500' in data['market_data']:
            fig = px.line(data['market_data']['SP500'], x='Date', y='Close', 
                         title='S&P 500 Historical Performance')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if 'NASDAQ' in data['market_data']:
            fig = px.line(data['market_data']['NASDAQ'], x='Date', y='Close', 
                         title='NASDAQ Historical Performance')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if 'DowJones' in data['market_data']:
            fig = px.line(data['market_data']['DowJones'], x='Date', y='Close', 
                         title='Dow Jones Historical Performance')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        if 'Volatility' in data['market_data']:
            fig = px.line(data['market_data']['Volatility'], x='Date', y='Close', 
                         title='Market Volatility (VIX)')
            st.plotly_chart(fig, use_container_width=True)
    
    # Industry Risk Analysis
    st.header("Industry Risk Analysis")
    if data['industry_risk']:
        industry_df = pd.DataFrame(data['industry_risk']).T
        fig = px.bar(industry_df, x=industry_df.index, y='default_rate',
                    title='Industry Default Rates')
        st.plotly_chart(fig, use_container_width=True)
    
    # Regulatory Framework
    st.header("Regulatory Framework")
    if data['regulatory_data']:
        regulatory_df = pd.DataFrame(data['regulatory_data']).T
        fig = px.bar(regulatory_df, x=regulatory_df.index, y='regulatory_quality',
                    title='Regulatory Quality by Country')
        st.plotly_chart(fig, use_container_width=True)
    
    # Economic Indicators
    st.header("Economic Indicators")
    if 'gdp' in data['economic_data']:
        gdp_data = data['economic_data']['gdp']
        fig = px.line(gdp_data, x='Year', y='GDP_per_capita', 
                     title='GDP per Capita Trends')
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk Score Distribution
    st.header("Risk Score Distribution")
    # Generate sample risk scores for demonstration
    risk_scores = np.random.normal(600, 100, 1000)
    fig = px.histogram(x=risk_scores, nbins=50, 
                      title='Distribution of Risk Scores',
                      labels={'x': 'Risk Score'},
                      color_discrete_sequence=['#1f77b4'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk Categories
    st.header("Risk Categories")
    col1, col2 = st.columns(2)
    
    with col1:
        risk_categories = pd.cut(risk_scores, 
                               bins=[0, 500, 650, 800, 1000],
                               labels=['Very High Risk', 'High Risk', 'Medium Risk', 'Low Risk'])
        risk_counts = pd.Series(risk_categories).value_counts()
        fig = px.pie(values=risk_counts.values, 
                    names=risk_counts.index,
                    title='Portfolio Risk Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average metrics by risk category
        metrics = pd.DataFrame({
            'risk_category': risk_categories,
            'credit_limit': np.random.uniform(1000, 10000, 1000),
            'default_probability': np.random.uniform(0, 0.3, 1000)
        })
        metrics = metrics.groupby('risk_category').mean().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=metrics['risk_category'],
            y=metrics['credit_limit'],
            name='Average Credit Limit'
        ))
        fig.add_trace(go.Bar(
            x=metrics['risk_category'],
            y=metrics['default_probability'] * 100,
            name='Default Probability (%)'
        ))
        fig.update_layout(
            title='Average Metrics by Risk Category',
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Failed to load data. Please check your data files and try again.") 