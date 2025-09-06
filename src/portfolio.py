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
    
    def get_total_value(self) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash_balance + positions_value
    
    def get_position_weights(self) -> Dict[str, float]:
        """Calculate current position weights."""
        total_value = self.get_total_value()
        if total_value <= 0:
            return {}
        
        weights = {}
        for symbol, position in self.positions.items():
            weights[symbol] = position.market_value / total_value
        
        return weights
    
    def rebalance_to_targets(self, target_weights: Dict[str, float], prices: Dict[str, float], date: datetime):
        """
        Rebalance portfolio to target weights.
        
        Args:
            target_weights: Dictionary of symbol -> target weight
            prices: Current prices
            date: Rebalance date
        """
        logger.info(f"Rebalancing portfolio on {date.strftime('%Y-%m-%d')}")
        
        # Update current prices
        self.update_prices(prices, date)
        
        # Calculate current portfolio value
        total_value = self.get_total_value()
        
        if total_value <= 0:
            logger.error("Portfolio value is zero or negative, cannot rebalance")
            return
        
        # Calculate target values and required trades
        trades = {}
        for symbol, target_weight in target_weights.items():
            if symbol not in prices:
                logger.warning(f"No price data for {symbol}, skipping")
                continue
                
            target_value = total_value * target_weight
            current_value = self.positions.get(symbol, Position(symbol, 0, 0)).market_value
            trade_value = target_value - current_value
            
            if abs(trade_value) > 1.0:  # Minimum trade size
                trade_quantity = trade_value / prices[symbol]
                trades[symbol] = trade_quantity
        
        # Execute trades
        for symbol, quantity in trades.items():
            self.add_position(symbol, quantity, prices[symbol], date)
    
    def record_performance(self, date: datetime):
        """Record current portfolio state for performance tracking."""
        performance_record = {
            'date': date,
            'total_value': self.get_total_value(),
            'cash_balance': self.cash_balance,
            'positions': {symbol: {
                'quantity': pos.quantity,
                'avg_price': pos.avg_price,
                'current_price': pos.current_price,
                'market_value': pos.market_value
            } for symbol, pos in self.positions.items()}
        }
        
        self.performance_history.append(performance_record)
    
    def get_performance_metrics(self, benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate basic performance metrics.
        
        Args:
            benchmark_returns: Benchmark returns for comparison
            
        Returns:
            Dictionary of performance metrics
        """
        if len(self.performance_history) < 2:
            return {}
        
        # Convert to DataFrame for easier calculation
        perf_df = pd.DataFrame(self.performance_history)
        perf_df['date'] = pd.to_datetime(perf_df['date'])
        perf_df = perf_df.set_index('date').sort_index()
        
        # Calculate returns
        perf_df['portfolio_return'] = perf_df['total_value'].pct_change()
        
        # Basic metrics
        total_return = (perf_df['total_value'].iloc[-1] / perf_df['total_value'].iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(perf_df)) - 1
        volatility = perf_df['portfolio_return'].std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        running_max = perf_df['total_value'].expanding().max()
        drawdown = (perf_df['total_value'] - running_max) / running_max
        max_drawdown = drawdown.min()
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trade_history)
        }
        
        return metrics