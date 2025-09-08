"""
Trading strategies module for quant fund simulation.

Implements momentum, mean-reversion, and other quantitative strategies.
Core backtesting engine for strategy evaluation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import warnings

from portfolio import Portfolio

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Defines the interface that all strategies must implement.
    """
    
    def __init__(self, name: str, lookback_period: int = 20):
        """
        Initialize base strategy.
        
        Args:
            name: Strategy name
            lookback_period: Default lookback period for calculations
        """
        self.name = name
        self.lookback_period = lookback_period
        self.signals = {}
        self.performance_history = []
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, date: datetime) -> Dict[str, float]:
        """
        Generate trading signals for given date.
        
        Args:
            data: Historical price data
            date: Current date
            
        Returns:
            Dictionary of symbol -> signal strength (-1 to 1)
        """
        pass
    
    def get_signal_strength(self, symbol: str, date: datetime) -> float:
        """Get signal strength for symbol on given date."""
        if date in self.signals and symbol in self.signals[date]:
            return self.signals[date][symbol]
        return 0.0
    
    def record_performance(self, date: datetime, portfolio_value: float):
        """Record strategy performance."""
        self.performance_history.append({
            'date': date,
            'portfolio_value': portfolio_value
        })


class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy based on price and return momentum.
    
    Buys assets with strong positive momentum, sells assets with negative momentum.
    """
    
    def __init__(self, 
                 name: str = "momentum",
                 lookback_period: int = 20,
                 momentum_threshold: float = 0.02,
                 rebalance_frequency: int = 5):
        """
        Initialize momentum strategy.
        
        Args:
            name: Strategy name
            lookback_period: Period for momentum calculation
            momentum_threshold: Minimum momentum to trigger signal
            rebalance_frequency: Days between rebalancing
        """
        super().__init__(name, lookback_period)
        self.momentum_threshold = momentum_threshold
        self.rebalance_frequency = rebalance_frequency
    
    def generate_signals(self, data: pd.DataFrame, date: datetime) -> Dict[str, float]:
        """
        Generate momentum signals.
        
        Signal calculation:
        1. Calculate price momentum over lookback period
        2. Calculate return momentum (recent vs historical returns)
        3. Combine signals with threshold filtering
        """
        signals = {}
        
        # Get available symbols from data columns
        symbols = [col.split('_')[0] for col in data.columns if 'Close' in col]
        
        for symbol in symbols:
            close_col = f"{symbol}_Close"
            if close_col not in data.columns:
                continue
            
            # Get price data up to current date
            price_data = data[close_col].loc[:date].dropna()
            
            if len(price_data) < self.lookback_period:
                signals[symbol] = 0.0
                continue
            
            # Calculate price momentum
            current_price = price_data.iloc[-1]
            past_price = price_data.iloc[-self.lookback_period]
            price_momentum = (current_price - past_price) / past_price
            
            # Calculate return momentum (recent vs historical volatility)
            returns = price_data.pct_change().dropna()
            if len(returns) < self.lookback_period:
                signals[symbol] = 0.0
                continue
            
            recent_returns = returns.tail(5).mean()
            historical_vol = returns.std()
            
            # Normalize return momentum
            return_momentum = recent_returns / historical_vol if historical_vol > 0 else 0
            
            # Combine signals
            combined_signal = (price_momentum + return_momentum) / 2
            
            # Apply threshold
            if abs(combined_signal) < self.momentum_threshold:
                signals[symbol] = 0.0
            else:
                # Normalize to [-1, 1] range
                signals[symbol] = np.clip(combined_signal / self.momentum_threshold, -1, 1)
        
        # Store signals
        self.signals[date] = signals
        
        return signals


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy based on price deviation from moving averages.
    
    Buys assets trading below their mean, sells assets trading above their mean.
    """
    
    def __init__(self, 
                 name: str = "mean_reversion",
                 lookback_period: int = 20,
                 deviation_threshold: float = 0.1,
                 rebalance_frequency: int = 5):
        """
        Initialize mean reversion strategy.
        
        Args:
            name: Strategy name
            lookback_period: Period for moving average calculation
            deviation_threshold: Minimum deviation to trigger signal
            rebalance_frequency: Days between rebalancing
        """
        super().__init__(name, lookback_period)
        self.deviation_threshold = deviation_threshold
        self.rebalance_frequency = rebalance_frequency
    
    def generate_signals(self, data: pd.DataFrame, date: datetime) -> Dict[str, float]:
        """
        Generate mean reversion signals.
        
        Signal calculation:
        1. Calculate moving average over lookback period
        2. Calculate deviation from moving average
        3. Generate contrarian signals (buy low, sell high)
        """
        signals = {}
        
        # Get available symbols from data columns
        symbols = [col.split('_')[0] for col in data.columns if 'Close' in col]
        
        for symbol in symbols:
            close_col = f"{symbol}_Close"
            if close_col not in data.columns:
                continue
            
            # Get price data up to current date
            price_data = data[close_col].loc[:date].dropna()
            
            if len(price_data) < self.lookback_period:
                signals[symbol] = 0.0
                continue
            
            # Calculate moving average
            current_price = price_data.iloc[-1]
            moving_avg = price_data.tail(self.lookback_period).mean()
            
            # Calculate deviation from mean
            deviation = (current_price - moving_avg) / moving_avg
            
            # Generate contrarian signal
            if abs(deviation) < self.deviation_threshold:
                signals[symbol] = 0.0
            else:
                # Negative signal for positive deviation (sell high)
                # Positive signal for negative deviation (buy low)
                signals[symbol] = -np.clip(deviation / self.deviation_threshold, -1, 1)
        
        # Store signals
        self.signals[date] = signals
        
        return signals


class BacktestEngine:
    """
    Backtesting engine for strategy evaluation.
    
    Handles portfolio simulation, signal execution, and performance tracking.
    """
    
    def __init__(self, 
                 initial_cash: float = 100000.0,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005):
        """
        Initialize backtesting engine.
        
        Args:
            initial_cash: Starting cash amount
            transaction_cost: Transaction cost as fraction of trade value
            slippage: Market impact as fraction of trade value
        """
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.portfolio = None
        self.benchmark_data = None
        
    def run_backtest(self, 
                    strategy: BaseStrategy,
                    data: pd.DataFrame,
                    start_date: datetime,
                    end_date: datetime,
                    rebalance_frequency: int = 5,
                    benchmark_symbol: str = '^GSPC') -> Dict:
        """
        Run backtest for given strategy.
        
        Args:
            strategy: Trading strategy to test
            data: Historical price data
            start_date: Backtest start date
            end_date: Backtest end date
            rebalance_frequency: Days between rebalancing
            benchmark_symbol: Benchmark symbol for comparison
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest for {strategy.name} from {start_date} to {end_date}")
        
        # Initialize portfolio
        self.portfolio = Portfolio(
            initial_cash=self.initial_cash,
            portfolio_id=f"{strategy.name}_backtest"
        )
        
        # Get benchmark data
        self.benchmark_data = self._get_benchmark_data(benchmark_symbol, start_date, end_date)
        
        # Get trading dates
        trading_dates = self._get_trading_dates(data, start_date, end_date)