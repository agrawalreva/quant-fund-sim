"""
Portfolio management module for quant fund simulation.

Handles portfolio construction, rebalancing, and performance tracking.
Core foundation for all trading strategies.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a single position in the portfolio."""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        """Calculate current market value of position."""
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        return (self.current_price - self.avg_price) * self.quantity
    
    @property
    def weight(self) -> float:
        """Calculate position weight (requires total portfolio value)."""
        return 0.0  # Will be calculated by portfolio


class Portfolio:
    """
    Core portfolio class for managing positions and tracking performance.
    
    Handles position management, cash balance, and basic performance metrics.
    """
    
    def __init__(self, 
                 initial_cash: float = 100000.0,
                 portfolio_id: str = "default",
                 rebalance_frequency: str = "monthly"):
        """
        Initialize portfolio.
        
        Args:
            initial_cash: Starting cash amount
            portfolio_id: Unique identifier for portfolio
            rebalance_frequency: How often to rebalance (daily, weekly, monthly)
        """
        self.portfolio_id = portfolio_id
        self.initial_cash = initial_cash
        self.cash_balance = initial_cash
        self.positions: Dict[str, Position] = {}
        self.rebalance_frequency = rebalance_frequency
        
        # Performance tracking
        self.performance_history = []
        self.trade_history = []
        self.current_date = None
        
        logger.info(f"Initialized portfolio {portfolio_id} with ${initial_cash:,.2f}")