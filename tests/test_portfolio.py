"""
Unit tests for portfolio module.

Tests core portfolio functionality including position management,
rebalancing, and performance tracking.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from portfolio import Portfolio, Position


class TestPosition:
    """Test Position class functionality."""
    
    def test_position_creation(self):
        """Test basic position creation."""
        pos = Position(symbol='AAPL', quantity=100, avg_price=150.0, current_price=160.0)
        
        assert pos.symbol == 'AAPL'
        assert pos.quantity == 100
        assert pos.avg_price == 150.0
        assert pos.current_price == 160.0
    
    def test_market_value(self):
        """Test market value calculation."""
        pos = Position(symbol='AAPL', quantity=100, avg_price=150.0, current_price=160.0)
        
        assert pos.market_value == 16000.0
    
    def test_unrealized_pnl(self):
        """Test unrealized P&L calculation."""
        pos = Position(symbol='AAPL', quantity=100, avg_price=150.0, current_price=160.0)
        
        assert pos.unrealized_pnl == 1000.0  # (160 - 150) * 100


class TestPortfolio:
    """Test Portfolio class functionality."""
    
    def setup_method(self):
        """Setup test portfolio for each test."""
        self.portfolio = Portfolio(initial_cash=100000, portfolio_id="test")
        self.date = datetime(2023, 1, 1)
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        assert self.portfolio.portfolio_id == "test"
        assert self.portfolio.initial_cash == 100000
        assert self.portfolio.cash_balance == 100000
        assert len(self.portfolio.positions) == 0
        assert len(self.portfolio.performance_history) == 0
    
    def test_add_position(self):
        """Test adding a position."""
        success = self.portfolio.add_position('AAPL', 100, 150.0, self.date)
        
        assert success is True
        assert 'AAPL' in self.portfolio.positions
        assert self.portfolio.positions['AAPL'].quantity == 100
        assert self.portfolio.positions['AAPL'].avg_price == 150.0
        assert self.portfolio.cash_balance == 85000  # 100000 - 15000
    
    def test_add_position_insufficient_cash(self):
        """Test adding position with insufficient cash."""
        success = self.portfolio.add_position('AAPL', 1000, 150.0, self.date)
        
        assert success is False
        assert len(self.portfolio.positions) == 0
        assert self.portfolio.cash_balance == 100000  # Unchanged
    
    def test_update_position(self):
        """Test updating existing position."""
        # Add initial position
        self.portfolio.add_position('AAPL', 100, 150.0, self.date)
        
        # Add more shares
        self.portfolio.add_position('AAPL', 50, 160.0, self.date)
        
        pos = self.portfolio.positions['AAPL']
        assert pos.quantity == 150
        assert pos.avg_price == 153.33  # (100*150 + 50*160) / 150
        assert self.portfolio.cash_balance == 75000  # 100000 - 15000 - 8000
    
    def test_close_position(self):
        """Test closing a position."""
        # Add position
        self.portfolio.add_position('AAPL', 100, 150.0, self.date)
        
        # Close position
        self.portfolio.add_position('AAPL', -100, 160.0, self.date)
        
        assert 'AAPL' not in self.portfolio.positions
        assert self.portfolio.cash_balance == 101000  # 100000 - 15000 + 16000
    
    def test_update_prices(self):
        """Test updating position prices."""
        # Add position
        self.portfolio.add_position('AAPL', 100, 150.0, self.date)
        
        # Update prices
        prices = {'AAPL': 160.0}
        self.portfolio.update_prices(prices, self.date)
        
        assert self.portfolio.positions['AAPL'].current_price == 160.0
        assert self.portfolio.current_date == self.date
    
    def test_get_total_value(self):
        """Test total portfolio value calculation."""
        # Add position
        self.portfolio.add_position('AAPL', 100, 150.0, self.date)
        
        # Update prices
        prices = {'AAPL': 160.0}
        self.portfolio.update_prices(prices, self.date)
        
        total_value = self.portfolio.get_total_value()
        expected_value = 85000 + 16000  # cash + position value
        assert total_value == expected_value
    
    def test_get_position_weights(self):
        """Test position weight calculation."""
        # Add positions
        self.portfolio.add_position('AAPL', 100, 150.0, self.date)
        self.portfolio.add_position('MSFT', 50, 300.0, self.date)
        
        # Update prices
        prices = {'AAPL': 160.0, 'MSFT': 320.0}
        self.portfolio.update_prices(prices, self.date)
        
        weights = self.portfolio.get_position_weights()
        
        total_value = self.portfolio.get_total_value()
        expected_aapl_weight = 16000 / total_value
        expected_msft_weight = 16000 / total_value
        
        assert abs(weights['AAPL'] - expected_aapl_weight) < 0.001
        assert abs(weights['MSFT'] - expected_msft_weight) < 0.001
    
    def test_rebalance_to_targets(self):
        """Test portfolio rebalancing."""
        # Add initial positions
        self.portfolio.add_position('AAPL', 100, 150.0, self.date)
        self.portfolio.add_position('MSFT', 50, 300.0, self.date)
        
        # Update prices
        prices = {'AAPL': 160.0, 'MSFT': 320.0, 'GOOGL': 2500.0}
        self.portfolio.update_prices(prices, self.date)
        
        # Rebalance to new targets
        target_weights = {'AAPL': 0.3, 'MSFT': 0.3, 'GOOGL': 0.4}
        self.portfolio.rebalance_to_targets(target_weights, prices, self.date)
        
        # Check that positions exist
        assert 'AAPL' in self.portfolio.positions
        assert 'MSFT' in self.portfolio.positions
        assert 'GOOGL' in self.portfolio.positions
    
    def test_record_performance(self):
        """Test performance recording."""
        # Add position and update prices
        self.portfolio.add_position('AAPL', 100, 150.0, self.date)
        prices = {'AAPL': 160.0}
        self.portfolio.update_prices(prices, self.date)
        
        # Record performance
        self.portfolio.record_performance(self.date)
        
        assert len(self.portfolio.performance_history) == 1
        perf_record = self.portfolio.performance_history[0]
        
        assert perf_record['date'] == self.date
        assert perf_record['total_value'] == self.portfolio.get_total_value()
        assert perf_record['cash_balance'] == self.portfolio.cash_balance
        assert 'AAPL' in perf_record['positions']
    
    def test_get_performance_metrics(self):
        """Test performance metrics calculation."""
        # Create some performance history
        dates = [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)]
        values = [100000, 101000, 102000]
        
        for date, value in zip(dates, values):
            # Simulate portfolio value
            self.portfolio.cash_balance = value
            self.portfolio.record_performance(date)
        
        metrics = self.portfolio.get_performance_metrics()
        
        assert 'total_return' in metrics
        assert 'annualized_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
    
    def test_serialization(self):
        """Test portfolio serialization and deserialization."""
        # Add some data
        self.portfolio.add_position('AAPL', 100, 150.0, self.date)
        self.portfolio.record_performance(self.date)
        
        # Serialize
        portfolio_dict = self.portfolio.to_dict()
        
        # Deserialize
        new_portfolio = Portfolio.from_dict(portfolio_dict)
        
        assert new_portfolio.portfolio_id == self.portfolio.portfolio_id
        assert new_portfolio.cash_balance == self.portfolio.cash_balance
        assert len(new_portfolio.positions) == len(self.portfolio.positions)
        assert len(new_portfolio.performance_history) == len(self.portfolio.performance_history)


if __name__ == "__main__":
    pytest.main([__file__])
