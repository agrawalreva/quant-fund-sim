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