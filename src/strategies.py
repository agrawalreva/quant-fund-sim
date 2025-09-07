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