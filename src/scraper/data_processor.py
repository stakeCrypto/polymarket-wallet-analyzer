"""Data processing and normalization utilities."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional

import pandas as pd

from ..config import get_logger
from .wallet_fetcher import Position, Redemption, Trade

logger = get_logger(__name__)


@dataclass
class ProcessedTrade:
    """Trade with computed P&L information."""

    trade: Trade
    realized_pnl: Decimal = Decimal("0")
    is_closing: bool = False
    position_size_after: Decimal = Decimal("0")


@dataclass
class MarketSummary:
    """Summary statistics for a single market."""

    market_id: str
    market_title: str
    category: str
    total_volume: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")
    trade_count: int = 0
    win_count: int = 0
    first_trade: Optional[datetime] = None
    last_trade: Optional[datetime] = None
    avg_hold_time_hours: Optional[float] = None

    @property
    def win_rate(self) -> float:
        """Calculate win rate for this market."""
        if self.trade_count == 0:
            return 0.0
        return self.win_count / self.trade_count


@dataclass
class PositionTracker:
    """Tracks position state for P&L calculation."""

    market_id: str
    outcome: str
    size: Decimal = Decimal("0")
    total_cost: Decimal = Decimal("0")
    trades: list[Trade] = field(default_factory=list)

    @property
    def avg_price(self) -> Decimal:
        """Calculate average entry price."""
        if self.size == 0:
            return Decimal("0")
        return self.total_cost / self.size

    def add_buy(self, trade: Trade) -> Decimal:
        """Process a buy trade, return realized P&L (always 0 for buys).

        Args:
            trade: The buy trade to process.

        Returns:
            Realized P&L (0 for buys).
        """
        self.size += trade.size
        self.total_cost += trade.cost_usd
        self.trades.append(trade)
        return Decimal("0")

    def add_sell(self, trade: Trade) -> Decimal:
        """Process a sell trade, return realized P&L.

        Args:
            trade: The sell trade to process.

        Returns:
            Realized P&L from closing portion of position.
        """
        if self.size == 0:
            # Short selling not tracked
            return Decimal("0")

        # Calculate P&L on sold portion
        sell_size = min(trade.size, self.size)
        cost_basis = self.avg_price * sell_size
        sale_proceeds = trade.price * sell_size
        realized_pnl = sale_proceeds - cost_basis

        # Update position
        self.size -= sell_size
        if self.size > 0:
            self.total_cost = self.avg_price * self.size
        else:
            self.total_cost = Decimal("0")

        self.trades.append(trade)

        return realized_pnl


class DataProcessor:
    """Processes and enriches trading data."""

    def __init__(self) -> None:
        """Initialize the data processor."""
        self.position_trackers: dict[str, PositionTracker] = {}

    def _get_position_key(self, market_id: str, outcome: str) -> str:
        """Generate unique key for position tracking."""
        return f"{market_id}:{outcome}"

    def _get_or_create_tracker(self, market_id: str, outcome: str) -> PositionTracker:
        """Get or create a position tracker."""
        key = self._get_position_key(market_id, outcome)
        if key not in self.position_trackers:
            self.position_trackers[key] = PositionTracker(
                market_id=market_id,
                outcome=outcome
            )
        return self.position_trackers[key]

    def calculate_realized_pnl(self, trades: list[Trade]) -> list[ProcessedTrade]:
        """Calculate realized P&L for each trade.

        Uses FIFO (First In, First Out) accounting.

        Args:
            trades: List of trades sorted by timestamp.

        Returns:
            List of processed trades with P&L information.
        """
        self.position_trackers.clear()
        processed: list[ProcessedTrade] = []

        for trade in trades:
            tracker = self._get_or_create_tracker(trade.market_id, trade.outcome)

            if trade.is_buy:
                realized_pnl = tracker.add_buy(trade)
                is_closing = False
            else:
                realized_pnl = tracker.add_sell(trade)
                is_closing = True

            processed.append(ProcessedTrade(
                trade=trade,
                realized_pnl=realized_pnl,
                is_closing=is_closing,
                position_size_after=tracker.size
            ))

        return processed

    def calculate_redemption_pnl(
        self,
        trades: list[Trade],
        redemptions: list[Redemption]
    ) -> tuple[Decimal, list[dict]]:
        """Calculate P&L from market redemptions (resolution payouts).

        For each redemption, finds the original buy trades and calculates
        the profit/loss based on cost basis vs redemption payout.

        Args:
            trades: List of trades to find cost basis from.
            redemptions: List of redemptions (market resolution payouts).

        Returns:
            Tuple of (total_redemption_pnl, list of redemption details).
        """
        if not redemptions:
            return Decimal("0"), []

        # Build cost basis by market
        market_costs: dict[str, dict] = defaultdict(lambda: {
            "total_cost": Decimal("0"),
            "total_size": Decimal("0"),
            "buys": []
        })

        for trade in trades:
            if trade.is_buy:
                key = trade.market_id
                market_costs[key]["total_cost"] += trade.cost_usd
                market_costs[key]["total_size"] += trade.size
                market_costs[key]["buys"].append(trade)

        # Calculate P&L for each redemption
        total_pnl = Decimal("0")
        redemption_details = []

        for redemption in redemptions:
            market_id = redemption.market_id
            cost_data = market_costs.get(market_id)

            if cost_data and cost_data["total_size"] > 0:
                # Calculate average cost per share
                avg_cost = cost_data["total_cost"] / cost_data["total_size"]

                # Cost basis for redeemed shares
                # Note: redemption.size is shares redeemed, usdc_received is payout
                redeemed_size = min(redemption.size, cost_data["total_size"])
                cost_basis = avg_cost * redeemed_size

                # P&L = payout - cost basis
                pnl = redemption.usdc_received - cost_basis
            else:
                # No buys found, treat redemption as pure profit
                pnl = redemption.usdc_received
                cost_basis = Decimal("0")

            total_pnl += pnl

            redemption_details.append({
                "market_id": market_id,
                "market_title": redemption.market_title,
                "size_redeemed": float(redemption.size),
                "usdc_received": float(redemption.usdc_received),
                "cost_basis": float(cost_basis) if cost_data else 0,
                "pnl": float(pnl),
                "timestamp": redemption.timestamp.isoformat()
            })

        logger.info(f"Calculated redemption P&L: ${total_pnl:.2f} from {len(redemptions)} redemptions")
        return total_pnl, redemption_details

    def compute_market_summaries(
        self,
        processed_trades: list[ProcessedTrade]
    ) -> dict[str, MarketSummary]:
        """Compute summary statistics by market.

        Args:
            processed_trades: List of processed trades.

        Returns:
            Dictionary mapping market_id to MarketSummary.
        """
        summaries: dict[str, MarketSummary] = {}

        # Group trades by market
        market_trades: dict[str, list[ProcessedTrade]] = defaultdict(list)
        for pt in processed_trades:
            market_trades[pt.trade.market_id].append(pt)

        for market_id, trades in market_trades.items():
            if not trades:
                continue

            first_trade = trades[0].trade
            summary = MarketSummary(
                market_id=market_id,
                market_title=first_trade.market_title,
                category=first_trade.market_category
            )

            # Calculate stats
            total_pnl = Decimal("0")
            closing_trades = []

            for pt in trades:
                summary.total_volume += pt.trade.cost_usd
                summary.trade_count += 1
                total_pnl += pt.realized_pnl

                if pt.is_closing and pt.realized_pnl > 0:
                    summary.win_count += 1
                    closing_trades.append(pt)

            summary.total_pnl = total_pnl
            summary.first_trade = trades[0].trade.timestamp
            summary.last_trade = trades[-1].trade.timestamp

            # Calculate average hold time
            if closing_trades:
                hold_times = []
                for ct in closing_trades:
                    # Find opening trade
                    for pt in trades:
                        if (pt.trade.is_buy and
                            pt.trade.outcome == ct.trade.outcome and
                            pt.trade.timestamp < ct.trade.timestamp):
                            hold_time = (ct.trade.timestamp - pt.trade.timestamp).total_seconds() / 3600
                            hold_times.append(hold_time)
                            break

                if hold_times:
                    summary.avg_hold_time_hours = sum(hold_times) / len(hold_times)

            summaries[market_id] = summary

        return summaries

    def to_dataframe(self, trades: list[Trade]) -> pd.DataFrame:
        """Convert trades to pandas DataFrame.

        Args:
            trades: List of Trade objects.

        Returns:
            DataFrame with trade data.
        """
        data = []
        for trade in trades:
            data.append({
                "id": trade.id,
                "wallet_address": trade.wallet_address,
                "market_id": trade.market_id,
                "market_title": trade.market_title,
                "market_category": trade.market_category,
                "outcome": trade.outcome,
                "side": trade.side,
                "price": float(trade.price),
                "size": float(trade.size),
                "cost_usd": float(trade.cost_usd),
                "timestamp": trade.timestamp,
                "transaction_hash": trade.transaction_hash,
                "fee_usd": float(trade.fee_usd)
            })

        df = pd.DataFrame(data)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    def processed_to_dataframe(
        self,
        processed_trades: list[ProcessedTrade]
    ) -> pd.DataFrame:
        """Convert processed trades to pandas DataFrame.

        Args:
            processed_trades: List of ProcessedTrade objects.

        Returns:
            DataFrame with processed trade data including P&L.
        """
        data = []
        for pt in processed_trades:
            trade = pt.trade
            data.append({
                "id": trade.id,
                "wallet_address": trade.wallet_address,
                "market_id": trade.market_id,
                "market_title": trade.market_title,
                "market_category": trade.market_category,
                "outcome": trade.outcome,
                "side": trade.side,
                "price": float(trade.price),
                "size": float(trade.size),
                "cost_usd": float(trade.cost_usd),
                "timestamp": trade.timestamp,
                "realized_pnl": float(pt.realized_pnl),
                "is_closing": pt.is_closing,
                "position_size_after": float(pt.position_size_after),
                "cumulative_pnl": 0.0  # Will be calculated below
            })

        df = pd.DataFrame(data)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
            df["cumulative_pnl"] = df["realized_pnl"].cumsum()

        return df

    def aggregate_by_period(
        self,
        df: pd.DataFrame,
        period: str = "D"
    ) -> pd.DataFrame:
        """Aggregate trade data by time period.

        Args:
            df: DataFrame with trade data.
            period: Pandas period string (D=daily, W=weekly, M=monthly).

        Returns:
            Aggregated DataFrame.
        """
        if df.empty:
            return pd.DataFrame()

        df_agg = df.set_index("timestamp").resample(period).agg({
            "cost_usd": "sum",
            "realized_pnl": "sum",
            "id": "count"
        }).rename(columns={
            "cost_usd": "volume",
            "realized_pnl": "pnl",
            "id": "trade_count"
        })

        df_agg["cumulative_pnl"] = df_agg["pnl"].cumsum()

        return df_agg.reset_index()

    def compute_category_stats(
        self,
        processed_trades: list[ProcessedTrade]
    ) -> dict[str, dict[str, Any]]:
        """Compute statistics grouped by market category.

        Args:
            processed_trades: List of processed trades.

        Returns:
            Dictionary mapping category to statistics.
        """
        category_data: dict[str, dict[str, Any]] = defaultdict(lambda: {
            "volume": Decimal("0"),
            "pnl": Decimal("0"),
            "trade_count": 0,
            "win_count": 0
        })

        for pt in processed_trades:
            cat = pt.trade.market_category
            category_data[cat]["volume"] += pt.trade.cost_usd
            category_data[cat]["pnl"] += pt.realized_pnl
            category_data[cat]["trade_count"] += 1

            if pt.is_closing and pt.realized_pnl > 0:
                category_data[cat]["win_count"] += 1

        # Calculate win rates
        result = {}
        for cat, data in category_data.items():
            closing_count = sum(
                1 for pt in processed_trades
                if pt.trade.market_category == cat and pt.is_closing
            )

            result[cat] = {
                "volume": float(data["volume"]),
                "pnl": float(data["pnl"]),
                "trade_count": data["trade_count"],
                "win_rate": data["win_count"] / closing_count if closing_count > 0 else 0.0
            }

        return result
