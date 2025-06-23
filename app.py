import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
import math

# Set page config
st.set_page_config(
    page_title="Black-Scholes Option Pricing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Black-Scholes Formula Functions
def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price"""
    if T <= 0:
        return max(S - K, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price"""
    if T <= 0:
        return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate option Greeks"""
    if T <= 0:
        return {
            'delta': 1.0 if option_type == 'call' and S > K else (0.0 if option_type == 'call' else -1.0 if S < K else 0.0),
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    # Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta
    if option_type == 'call':
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    # Vega
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    # Rho
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Option Calculator", "Interactive Curves", "Greeks Analysis"])

if page == "Option Calculator":
    st.title("📊 Black-Scholes Option Pricing Calculator")
    st.markdown("Calculate option prices using the Black-Scholes formula")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Parameters")
        S = st.number_input("Current Stock Price (S)", value=100.0, min_value=0.01, step=1.0)
        K = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, step=1.0)
        T = st.number_input("Time to Expiration (T) in years", value=1.0, min_value=0.001, step=0.1)
        r = st.number_input("Risk-free Rate (r) %", value=5.0, min_value=0.0, max_value=100.0, step=0.1) / 100
        sigma = st.number_input("Volatility (σ) %", value=20.0, min_value=0.1, max_value=1000.0, step=1.0) / 100
        
        option_type = st.selectbox("Option Type", ["Call", "Put"])
    
    with col2:
        st.subheader("Results")
        
        if option_type == "Call":
            price = black_scholes_call(S, K, T, r, sigma)
            greeks = calculate_greeks(S, K, T, r, sigma, 'call')
        else:
            price = black_scholes_put(S, K, T, r, sigma)
            greeks = calculate_greeks(S, K, T, r, sigma, 'put')
        
        st.metric("Option Price", f"${price:.4f}")
        
        st.subheader("Greeks")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Delta (Δ)", f"{greeks['delta']:.4f}")
            st.metric("Gamma (Γ)", f"{greeks['gamma']:.4f}")
            st.metric("Theta (Θ)", f"{greeks['theta']:.4f}")
        with col_b:
            st.metric("Vega (ν)", f"{greeks['vega']:.4f}")
            st.metric("Rho (ρ)", f"{greeks['rho']:.4f}")
    
    # Add explanation
    st.markdown("---")
    st.subheader("About the Black-Scholes Model")
    st.markdown("""
    The Black-Scholes model is used to calculate the theoretical price of options. The key assumptions are:
    - Stock prices follow a geometric Brownian motion
    - Constant risk-free interest rate
    - Constant volatility
    - No dividends during the option's life
    - European-style exercise
    """)

elif page == "Interactive Curves":
    st.title("📈 Interactive Option Pricing Curves")
    st.markdown("Visualize how option prices change with different parameters")
    
    # Parameter controls
    st.sidebar.subheader("Base Parameters")
    base_S = st.sidebar.slider("Stock Price", 50, 200, 100)
    base_K = st.sidebar.slider("Strike Price", 50, 200, 100)
    base_T = st.sidebar.slider("Time to Expiration (years)", 0.1, 2.0, 1.0, 0.1)
    base_r = st.sidebar.slider("Risk-free Rate (%)", 0.0, 15.0, 5.0, 0.5) / 100
    base_sigma = st.sidebar.slider("Volatility (%)", 5.0, 100.0, 20.0, 5.0) / 100
    
    curve_type = st.selectbox("Select Curve Type", [
        "Stock Price vs Option Price",
        "Volatility vs Option Price", 
        "Time to Expiration vs Option Price",
        "Strike Price vs Option Price"
    ])
    
    option_type = st.selectbox("Option Type", ["Call", "Put"], key="curve_option_type")
    
    # Generate data for curves
    if curve_type == "Stock Price vs Option Price":
        x_range = np.linspace(base_S * 0.5, base_S * 1.5, 100)
        if option_type == "Call":
            y_values = [black_scholes_call(x, base_K, base_T, base_r, base_sigma) for x in x_range]
        else:
            y_values = [black_scholes_put(x, base_K, base_T, base_r, base_sigma) for x in x_range]
        x_label = "Stock Price"
        
    elif curve_type == "Volatility vs Option Price":
        x_range = np.linspace(0.05, 1.0, 100)
        if option_type == "Call":
            y_values = [black_scholes_call(base_S, base_K, base_T, base_r, x) for x in x_range]
        else:
            y_values = [black_scholes_put(base_S, base_K, base_T, base_r, x) for x in x_range]
        x_range = x_range * 100  # Convert to percentage
        x_label = "Volatility (%)"
        
    elif curve_type == "Time to Expiration vs Option Price":
        x_range = np.linspace(0.01, 2.0, 100)
        if option_type == "Call":
            y_values = [black_scholes_call(base_S, base_K, x, base_r, base_sigma) for x in x_range]
        else:
            y_values = [black_scholes_put(base_S, base_K, x, base_r, base_sigma) for x in x_range]
        x_label = "Time to Expiration (years)"
        
    else:  # Strike Price vs Option Price
        x_range = np.linspace(base_K * 0.5, base_K * 1.5, 100)
        if option_type == "Call":
            y_values = [black_scholes_call(base_S, x, base_T, base_r, base_sigma) for x in x_range]
        else:
            y_values = [black_scholes_put(base_S, x, base_T, base_r, base_sigma) for x in x_range]
        x_label = "Strike Price"
    
    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_values,
        mode='lines',
        name=f'{option_type} Option Price',
        line=dict(width=3, color='blue' if option_type == 'Call' else 'red')
    ))
    
    fig.update_layout(
        title=f'{option_type} Option: {curve_type}',
        xaxis_title=x_label,
        yaxis_title='Option Price ($)',
        hovermode='x unified',
        template='plotly_white',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add current point marker
    if curve_type == "Stock Price vs Option Price":
        current_x = base_S
        if option_type == "Call":
            current_y = black_scholes_call(base_S, base_K, base_T, base_r, base_sigma)
        else:
            current_y = black_scholes_put(base_S, base_K, base_T, base_r, base_sigma)
    elif curve_type == "Volatility vs Option Price":
        current_x = base_sigma * 100
        if option_type == "Call":
            current_y = black_scholes_call(base_S, base_K, base_T, base_r, base_sigma)
        else:
            current_y = black_scholes_put(base_S, base_K, base_T, base_r, base_sigma)
    elif curve_type == "Time to Expiration vs Option Price":
        current_x = base_T
        if option_type == "Call":
            current_y = black_scholes_call(base_S, base_K, base_T, base_r, base_sigma)
        else:
            current_y = black_scholes_put(base_S, base_K, base_T, base_r, base_sigma)
    else:
        current_x = base_K
        if option_type == "Call":
            current_y = black_scholes_call(base_S, base_K, base_T, base_r, base_sigma)
        else:
            current_y = black_scholes_put(base_S, base_K, base_T, base_r, base_sigma)
    
    st.markdown(f"**Current Point**: {x_label} = {current_x:.2f}, Option Price = ${current_y:.4f}")

elif page == "Greeks Analysis":
    st.title("🔍 Greeks Analysis")
    st.markdown("Analyze option sensitivities (Greeks) across different parameters")
    
    # Parameter controls
    col1, col2 = st.columns(2)
    with col1:
        S = st.slider("Stock Price", 50, 200, 100, key="greeks_S")
        K = st.slider("Strike Price", 50, 200, 100, key="greeks_K")
        T = st.slider("Time to Expiration (years)", 0.1, 2.0, 1.0, 0.1, key="greeks_T")
    with col2:
        r = st.slider("Risk-free Rate (%)", 0.0, 15.0, 5.0, 0.5, key="greeks_r") / 100
        sigma = st.slider("Volatility (%)", 5.0, 100.0, 20.0, 5.0, key="greeks_sigma") / 100
        option_type = st.selectbox("Option Type", ["Call", "Put"], key="greeks_option_type")
    
    # Calculate Greeks
    greeks = calculate_greeks(S, K, T, r, sigma, option_type.lower())
    
    # Display Greeks
    st.subheader("Current Greeks Values")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Delta (Δ)", f"{greeks['delta']:.4f}", 
                  help="Sensitivity to stock price changes")
    with col2:
        st.metric("Gamma (Γ)", f"{greeks['gamma']:.4f}", 
                  help="Rate of change of Delta")
    with col3:
        st.metric("Theta (Θ)", f"{greeks['theta']:.4f}", 
                  help="Time decay per day")
    with col4:
        st.metric("Vega (ν)", f"{greeks['vega']:.4f}", 
                  help="Sensitivity to volatility (per 1%)")
    with col5:
        st.metric("Rho (ρ)", f"{greeks['rho']:.4f}", 
                  help="Sensitivity to interest rate (per 1%)")
    
    # Greeks visualization
    st.subheader("Greeks vs Stock Price")
    
    # Generate data for Greeks vs Stock Price
    stock_range = np.linspace(S * 0.5, S * 1.5, 100)
    greeks_data = {
        'stock_price': stock_range,
        'delta': [],
        'gamma': [],
        'theta': [],
        'vega': [],
        'rho': []
    }
    
    for stock_price in stock_range:
        g = calculate_greeks(stock_price, K, T, r, sigma, option_type.lower())
        greeks_data['delta'].append(g['delta'])
        greeks_data['gamma'].append(g['gamma'])
        greeks_data['theta'].append(g['theta'])
        greeks_data['vega'].append(g['vega'])
        greeks_data['rho'].append(g['rho'])
    
    # Create subplots for Greeks
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Delta', 'Gamma', 'Theta', 'Vega', 'Rho', ''),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"type": "table"}]]
    )
    
    # Delta
    fig.add_trace(
        go.Scatter(x=stock_range, y=greeks_data['delta'], name='Delta', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Gamma
    fig.add_trace(
        go.Scatter(x=stock_range, y=greeks_data['gamma'], name='Gamma', line=dict(color='green')),
        row=1, col=2
    )
    
    # Theta
    fig.add_trace(
        go.Scatter(x=stock_range, y=greeks_data['theta'], name='Theta', line=dict(color='red')),
        row=1, col=3
    )
    
    # Vega
    fig.add_trace(
        go.Scatter(x=stock_range, y=greeks_data['vega'], name='Vega', line=dict(color='purple')),
        row=2, col=1
    )
    
    # Rho
    fig.add_trace(
        go.Scatter(x=stock_range, y=greeks_data['rho'], name='Rho', line=dict(color='orange')),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="Greeks Analysis")
    fig.update_xaxes(title_text="Stock Price", row=1, col=1)
    fig.update_xaxes(title_text="Stock Price", row=1, col=2)
    fig.update_xaxes(title_text="Stock Price", row=1, col=3)
    fig.update_xaxes(title_text="Stock Price", row=2, col=1)
    fig.update_xaxes(title_text="Stock Price", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Greeks explanation
    st.markdown("---")
    st.subheader("Greeks Explanation")
    st.markdown("""
    - **Delta (Δ)**: Measures the rate of change of option price with respect to stock price
    - **Gamma (Γ)**: Measures the rate of change of Delta with respect to stock price
    - **Theta (Θ)**: Measures the rate of change of option price with respect to time (time decay)
    - **Vega (ν)**: Measures the rate of change of option price with respect to volatility
    - **Rho (ρ)**: Measures the rate of change of option price with respect to interest rate
    """)

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit - Black-Scholes Option Pricing Tool*")