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
    
    def update_prices(self, prices: Dict[str, float], date: datetime):
        """
        Update current prices for all positions.
        
        Args:
            prices: Dictionary of symbol -> current price
            date: Current date
        """
        self.current_date = date
        
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def add_position(self, symbol: str, quantity: float, price: float, date: datetime):
        """
        Add or update a position.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares (positive for long, negative for short)
            price: Execution price
            date: Trade date
        """
        trade_value = quantity * price
        
        # Check if we have enough cash for long positions
        if quantity > 0 and trade_value > self.cash_balance:
            logger.warning(f"Insufficient cash for {symbol} trade. Required: ${trade_value:,.2f}, Available: ${self.cash_balance:,.2f}")
            return False
        
        # Update or create position
        if symbol in self.positions:
            existing_pos = self.positions[symbol]
            total_quantity = existing_pos.quantity + quantity
            total_cost = (existing_pos.quantity * existing_pos.avg_price) + trade_value
            
            if total_quantity != 0:
                new_avg_price = total_cost / total_quantity
                existing_pos.quantity = total_quantity
                existing_pos.avg_price = new_avg_price
            else:
                # Position closed
                del self.positions[symbol]
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                current_price=price
            )
        
        # Update cash balance
        self.cash_balance -= trade_value
        
        # Record trade
        self.trade_history.append({
            'date': date,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'value': trade_value,
            'cash_after': self.cash_balance
        })
        
        logger.info(f"Trade: {quantity:+.0f} {symbol} @ ${price:.2f} (${trade_value:+,.2f})")
        return True