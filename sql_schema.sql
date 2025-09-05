-- SQL Schema for Quant Fund Simulation Database
-- PostgreSQL compatible

-- Create database (run this separately)
-- CREATE DATABASE quant_fund_sim;

-- Price data table
CREATE TABLE IF NOT EXISTS price_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    open_price DECIMAL(10, 4),
    high_price DECIMAL(10, 4),
    low_price DECIMAL(10, 4),
    close_price DECIMAL(10, 4),
    volume BIGINT,
    adj_close DECIMAL(10, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date)
);

-- Portfolio states table
CREATE TABLE IF NOT EXISTS portfolio_states (
    id SERIAL PRIMARY KEY,
    portfolio_id VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    total_value DECIMAL(15, 2),
    cash_balance DECIMAL(15, 2),
    positions JSONB, -- Store positions as JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(portfolio_id, date)
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    portfolio_id VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    total_return DECIMAL(10, 6),
    daily_return DECIMAL(10, 6),
    cumulative_return DECIMAL(10, 6),
    volatility DECIMAL(10, 6),
    sharpe_ratio DECIMAL(10, 6),
    max_drawdown DECIMAL(10, 6),
    var_95 DECIMAL(10, 6),
    cvar_95 DECIMAL(10, 6),
    beta DECIMAL(10, 6),
    alpha DECIMAL(10, 6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(portfolio_id, date)
);

-- Strategy signals table
CREATE TABLE IF NOT EXISTS strategy_signals (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    signal INTEGER, -- -1, 0, 1 for sell, hold, buy
    confidence DECIMAL(5, 4), -- Signal confidence 0-1
    features JSONB, -- Store feature values as JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(strategy_name, symbol, date)
);

-- Macro economic data table
CREATE TABLE IF NOT EXISTS macro_data (
    id SERIAL PRIMARY KEY,
    series_id VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    value DECIMAL(15, 6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(series_id, date)
);

-- Model predictions table
CREATE TABLE IF NOT EXISTS model_predictions (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    prediction DECIMAL(10, 6),
    actual_return DECIMAL(10, 6),
    features JSONB,
    model_version VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, symbol, date)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_price_data_symbol_date ON price_data(symbol, date);
CREATE INDEX IF NOT EXISTS idx_portfolio_states_portfolio_date ON portfolio_states(portfolio_id, date);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_portfolio_date ON performance_metrics(portfolio_id, date);
CREATE INDEX IF NOT EXISTS idx_strategy_signals_strategy_symbol_date ON strategy_signals(strategy_name, symbol, date);
CREATE INDEX IF NOT EXISTS idx_macro_data_series_date ON macro_data(series_id, date);
CREATE INDEX IF NOT EXISTS idx_model_predictions_model_symbol_date ON model_predictions(model_name, symbol, date);
