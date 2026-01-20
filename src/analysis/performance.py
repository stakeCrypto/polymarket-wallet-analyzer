"""Performance analysis and metrics calculation."""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..config import get_logger
from ..scraper.data_processor import ProcessedTrade
from ..scraper.wallet_fetcher import Trade

logger = get_logger(__name__)


@dataclass
class TradePerformance:
    """Performance data for a single trade or trade group."""

    trade_id: str
    market_title: str
    market_category: str
    entry_price: float
    exit_price: Optional[float]
    size: float
    pnl: float
    pnl_percent: float
    hold_time_hours: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trade_id": self.trade_id,
            "market_title": self.market_title,
            "market_category": self.market_category,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "size": self.size,
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
            "hold_time_hours": self.hold_time_hours,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None
        }


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a wallet."""

    # Volume metrics
    total_volume_usd: float = 0.0
    total_buy_volume: float = 0.0
    total_sell_volume: float = 0.0

    # P&L metrics
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0

    # Trade counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    break_even_trades: int = 0

    # Win rate
    win_rate: float = 0.0
    loss_rate: float = 0.0

    # Best/Worst trades
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    best_trade: Optional[TradePerformance] = None
    worst_trade: Optional[TradePerformance] = None

    # Averages
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade_size: float = 0.0
    avg_hold_time_hours: float = 0.0

    # Risk metrics
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0

    # ROI
    roi_percent: float = 0.0

    # Time range
    first_trade_date: Optional[datetime] = None
    last_trade_date: Optional[datetime] = None
    active_days: int = 0

    # Category breakdown
    category_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "volume": {
                "total_usd": self.total_volume_usd,
                "buy_volume": self.total_buy_volume,
                "sell_volume": self.total_sell_volume
            },
            "pnl": {
                "realized": self.total_realized_pnl,
                "unrealized": self.total_unrealized_pnl,
                "gross_profit": self.gross_profit,
                "gross_loss": self.gross_loss,
                "net": self.total_realized_pnl + self.total_unrealized_pnl
            },
            "trades": {
                "total": self.total_trades,
                "winning": self.winning_trades,
                "losing": self.losing_trades,
                "break_even": self.break_even_trades
            },
            "rates": {
                "win_rate": self.win_rate,
                "loss_rate": self.loss_rate
            },
            "best_worst": {
                "best_pnl": self.best_trade_pnl,
                "worst_pnl": self.worst_trade_pnl,
                "best_trade": self.best_trade.to_dict() if self.best_trade else None,
                "worst_trade": self.worst_trade.to_dict() if self.worst_trade else None
            },
            "averages": {
                "avg_win": self.avg_win,
                "avg_loss": self.avg_loss,
                "avg_trade_size": self.avg_trade_size,
                "avg_hold_time_hours": self.avg_hold_time_hours
            },
            "risk": {
                "profit_factor": self.profit_factor,
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown": self.max_drawdown,
                "max_drawdown_percent": self.max_drawdown_percent
            },
            "roi_percent": self.roi_percent,
            "time_range": {
                "first_trade": self.first_trade_date.isoformat() if self.first_trade_date else None,
                "last_trade": self.last_trade_date.isoformat() if self.last_trade_date else None,
                "active_days": self.active_days
            },
            "by_category": self.category_metrics
        }


class PerformanceAnalyzer:
    """Analyzes trading performance and calculates metrics."""

    def __init__(self) -> None:
        """Initialize the performance analyzer."""
        self.metrics = PerformanceMetrics()
        self._pnl_series: list[float] = []
        self._trade_performances: list[TradePerformance] = []

    def _calculate_sharpe_ratio(
        self,
        returns: list[float],
        risk_free_rate: float = 0.0,
        periods_per_year: float = 365.0
    ) -> float:
        """Calculate Sharpe ratio from returns.

        Args:
            returns: List of period returns.
            risk_free_rate: Annual risk-free rate.
            periods_per_year: Number of periods in a year.

        Returns:
            Annualized Sharpe ratio.
        """
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / periods_per_year)

        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns, ddof=1)

        if std_return == 0:
            return 0.0

        sharpe = (mean_return / std_return) * math.sqrt(periods_per_year)

        return float(sharpe)

    def _calculate_max_drawdown(self, cumulative_pnl: list[float]) -> tuple[float, float]:
        """Calculate maximum drawdown.

        Args:
            cumulative_pnl: List of cumulative P&L values.

        Returns:
            Tuple of (max_drawdown_usd, max_drawdown_percent).
        """
        if not cumulative_pnl:
            return 0.0, 0.0

        peak = cumulative_pnl[0]
        max_dd = 0.0
        max_dd_pct = 0.0

        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl

            drawdown = peak - pnl

            if drawdown > max_dd:
                max_dd = drawdown
                if peak > 0:
                    max_dd_pct = drawdown / peak

        return max_dd, max_dd_pct

    def _build_trade_performance(
        self,
        processed_trades: list[ProcessedTrade]
    ) -> list[TradePerformance]:
        """Build trade performance records from processed trades.

        Args:
            processed_trades: List of processed trades.

        Returns:
            List of TradePerformance objects.
        """
        performances: list[TradePerformance] = []

        # Group by market and outcome
        from collections import defaultdict
        trade_groups: dict[str, list[ProcessedTrade]] = defaultdict(list)

        for pt in processed_trades:
            key = f"{pt.trade.market_id}:{pt.trade.outcome}"
            trade_groups[key].append(pt)

        for key, trades in trade_groups.items():
            # Find entry and exit trades
            buys = [t for t in trades if t.trade.is_buy]
            sells = [t for t in trades if t.trade.is_sell]

            if not buys:
                continue

            # Calculate entry stats
            total_buy_cost = sum(float(t.trade.cost_usd) for t in buys)
            total_buy_size = sum(float(t.trade.size) for t in buys)
            entry_price = total_buy_cost / total_buy_size if total_buy_size > 0 else 0
            entry_time = min(t.trade.timestamp for t in buys)

            # Calculate exit stats if any sells
            if sells:
                total_sell_proceeds = sum(float(t.trade.cost_usd) for t in sells)
                total_sell_size = sum(float(t.trade.size) for t in sells)
                exit_price = total_sell_proceeds / total_sell_size if total_sell_size > 0 else 0
                exit_time = max(t.trade.timestamp for t in sells)
                pnl = sum(float(t.realized_pnl) for t in trades)
                hold_time = (exit_time - entry_time).total_seconds() / 3600
            else:
                exit_price = None
                exit_time = None
                pnl = 0.0
                hold_time = None

            # Calculate P&L percent
            pnl_percent = (pnl / total_buy_cost * 100) if total_buy_cost > 0 else 0

            perf = TradePerformance(
                trade_id=buys[0].trade.id,
                market_title=buys[0].trade.market_title,
                market_category=buys[0].trade.market_category,
                entry_price=entry_price,
                exit_price=exit_price,
                size=total_buy_size,
                pnl=pnl,
                pnl_percent=pnl_percent,
                hold_time_hours=hold_time,
                entry_time=entry_time,
                exit_time=exit_time
            )

            performances.append(perf)

        return performances

    def analyze(self, processed_trades: list[ProcessedTrade]) -> PerformanceMetrics:
        """Analyze trading performance from processed trades.

        Args:
            processed_trades: List of processed trades with P&L info.

        Returns:
            PerformanceMetrics object with all calculated metrics.
        """
        if not processed_trades:
            return self.metrics

        logger.info(f"Analyzing performance for {len(processed_trades)} trades")

        # Build trade performances
        self._trade_performances = self._build_trade_performance(processed_trades)

        # Volume metrics
        for pt in processed_trades:
            self.metrics.total_volume_usd += float(pt.trade.cost_usd)
            if pt.trade.is_buy:
                self.metrics.total_buy_volume += float(pt.trade.cost_usd)
            else:
                self.metrics.total_sell_volume += float(pt.trade.cost_usd)

        # P&L metrics
        closing_trades = [pt for pt in processed_trades if pt.is_closing]

        for pt in closing_trades:
            pnl = float(pt.realized_pnl)
            self.metrics.total_realized_pnl += pnl
            self._pnl_series.append(pnl)

            if pnl > 0:
                self.metrics.gross_profit += pnl
                self.metrics.winning_trades += 1
            elif pnl < 0:
                self.metrics.gross_loss += abs(pnl)
                self.metrics.losing_trades += 1
            else:
                self.metrics.break_even_trades += 1

        self.metrics.total_trades = len(closing_trades)

        # Win rate
        if self.metrics.total_trades > 0:
            self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades
            self.metrics.loss_rate = self.metrics.losing_trades / self.metrics.total_trades

        # Best/Worst trades
        if self._trade_performances:
            sorted_by_pnl = sorted(self._trade_performances, key=lambda x: x.pnl)
            self.metrics.best_trade = sorted_by_pnl[-1]
            self.metrics.worst_trade = sorted_by_pnl[0]
            self.metrics.best_trade_pnl = self.metrics.best_trade.pnl
            self.metrics.worst_trade_pnl = self.metrics.worst_trade.pnl

        # Averages
        if self.metrics.winning_trades > 0:
            self.metrics.avg_win = self.metrics.gross_profit / self.metrics.winning_trades

        if self.metrics.losing_trades > 0:
            self.metrics.avg_loss = self.metrics.gross_loss / self.metrics.losing_trades

        if closing_trades:
            self.metrics.avg_trade_size = self.metrics.total_volume_usd / len(processed_trades)

        hold_times = [
            tp.hold_time_hours for tp in self._trade_performances
            if tp.hold_time_hours is not None
        ]
        if hold_times:
            self.metrics.avg_hold_time_hours = sum(hold_times) / len(hold_times)

        # Risk metrics
        if self.metrics.gross_loss > 0:
            self.metrics.profit_factor = self.metrics.gross_profit / self.metrics.gross_loss

        # Sharpe ratio (using daily P&L)
        if self._pnl_series:
            self.metrics.sharpe_ratio = self._calculate_sharpe_ratio(self._pnl_series)

        # Max drawdown
        cumulative_pnl = []
        running_total = 0.0
        for pnl in self._pnl_series:
            running_total += pnl
            cumulative_pnl.append(running_total)

        if cumulative_pnl:
            self.metrics.max_drawdown, self.metrics.max_drawdown_percent = \
                self._calculate_max_drawdown(cumulative_pnl)

        # ROI
        if self.metrics.total_buy_volume > 0:
            self.metrics.roi_percent = (
                self.metrics.total_realized_pnl / self.metrics.total_buy_volume * 100
            )

        # Time range
        if processed_trades:
            timestamps = [pt.trade.timestamp for pt in processed_trades]
            self.metrics.first_trade_date = min(timestamps)
            self.metrics.last_trade_date = max(timestamps)

            unique_days = len(set(t.date() for t in timestamps))
            self.metrics.active_days = unique_days

        # Category breakdown
        self.metrics.category_metrics = self._calculate_category_metrics(processed_trades)

        logger.info(
            f"Analysis complete: {self.metrics.total_trades} trades, "
            f"P&L: ${self.metrics.total_realized_pnl:.2f}, "
            f"Win Rate: {self.metrics.win_rate:.1%}"
        )

        return self.metrics

    def _calculate_category_metrics(
        self,
        processed_trades: list[ProcessedTrade]
    ) -> dict[str, dict[str, float]]:
        """Calculate metrics broken down by category.

        Args:
            processed_trades: List of processed trades.

        Returns:
            Dictionary mapping category to metrics.
        """
        from collections import defaultdict

        category_data: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "volume": 0.0,
                "pnl": 0.0,
                "trades": 0,
                "wins": 0
            }
        )

        for pt in processed_trades:
            cat = pt.trade.market_category
            category_data[cat]["volume"] += float(pt.trade.cost_usd)

            if pt.is_closing:
                category_data[cat]["pnl"] += float(pt.realized_pnl)
                category_data[cat]["trades"] += 1
                if pt.realized_pnl > 0:
                    category_data[cat]["wins"] += 1

        result: dict[str, dict[str, float]] = {}
        for cat, data in category_data.items():
            result[cat] = {
                "volume": data["volume"],
                "pnl": data["pnl"],
                "trade_count": data["trades"],
                "win_rate": data["wins"] / data["trades"] if data["trades"] > 0 else 0.0,
                "roi_percent": (data["pnl"] / data["volume"] * 100) if data["volume"] > 0 else 0.0
            }

        return result

    def get_top_trades(
        self,
        n: int = 10,
        by: str = "pnl"
    ) -> list[TradePerformance]:
        """Get top N trades by specified metric.

        Args:
            n: Number of trades to return.
            by: Metric to sort by (pnl, pnl_percent, size).

        Returns:
            List of top TradePerformance objects.
        """
        if not self._trade_performances:
            return []

        sorted_trades = sorted(
            self._trade_performances,
            key=lambda x: getattr(x, by, 0) or 0,
            reverse=True
        )

        return sorted_trades[:n]

    def get_worst_trades(
        self,
        n: int = 10,
        by: str = "pnl"
    ) -> list[TradePerformance]:
        """Get worst N trades by specified metric.

        Args:
            n: Number of trades to return.
            by: Metric to sort by (pnl, pnl_percent, size).

        Returns:
            List of worst TradePerformance objects.
        """
        if not self._trade_performances:
            return []

        sorted_trades = sorted(
            self._trade_performances,
            key=lambda x: getattr(x, by, 0) or 0
        )

        return sorted_trades[:n]

    def get_performance_by_period(
        self,
        processed_trades: list[ProcessedTrade],
        period: str = "M"
    ) -> pd.DataFrame:
        """Get performance metrics aggregated by time period.

        Args:
            processed_trades: List of processed trades.
            period: Pandas period string (D, W, M, Q, Y).

        Returns:
            DataFrame with period performance metrics.
        """
        if not processed_trades:
            return pd.DataFrame()

        data = []
        for pt in processed_trades:
            if pt.is_closing:
                data.append({
                    "timestamp": pt.trade.timestamp,
                    "pnl": float(pt.realized_pnl),
                    "volume": float(pt.trade.cost_usd)
                })

        df = pd.DataFrame(data)

        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        agg_df = df.resample(period).agg({
            "pnl": "sum",
            "volume": "sum"
        })

        agg_df["trade_count"] = df.resample(period).size()
        agg_df["cumulative_pnl"] = agg_df["pnl"].cumsum()
        agg_df["roi_percent"] = agg_df.apply(
            lambda r: (r["pnl"] / r["volume"] * 100) if r["volume"] > 0 else 0,
            axis=1
        )

        return agg_df.reset_index()
