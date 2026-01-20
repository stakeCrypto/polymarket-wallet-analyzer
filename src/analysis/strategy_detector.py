"""Strategy detection through pattern recognition."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from ..config import get_logger
from ..scraper.data_processor import ProcessedTrade
from ..scraper.wallet_fetcher import Trade

logger = get_logger(__name__)


class TradingStrategy(Enum):
    """Detected trading strategy types."""

    EARLY_EXIT = "early_exit"  # Sells before market resolution
    CONTRARIAN = "contrarian"  # Buys underdog positions
    MOMENTUM = "momentum"  # Follows price trends
    ARBITRAGE = "arbitrage"  # Opposite positions for guaranteed profit
    EVENT_DRIVEN = "event_driven"  # Trades around news/events
    SCALPING = "scalping"  # Quick small profit trades
    SWING = "swing"  # Medium-term position holding
    UNKNOWN = "unknown"


@dataclass
class StrategySignal:
    """A detected strategy signal from trade analysis."""

    strategy: TradingStrategy
    confidence: float  # 0.0 to 1.0
    evidence: list[str]
    sample_trades: list[str]  # Trade IDs


@dataclass
class StrategyProfile:
    """Complete strategy profile for a wallet."""

    primary_strategy: TradingStrategy
    secondary_strategies: list[TradingStrategy]
    signals: list[StrategySignal]
    behavior_traits: dict[str, Any]
    risk_profile: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "primary_strategy": self.primary_strategy.value,
            "secondary_strategies": [s.value for s in self.secondary_strategies],
            "signals": [
                {
                    "strategy": s.strategy.value,
                    "confidence": s.confidence,
                    "evidence": s.evidence,
                    "sample_count": len(s.sample_trades)
                }
                for s in self.signals
            ],
            "behavior_traits": self.behavior_traits,
            "risk_profile": self.risk_profile
        }


class StrategyDetector:
    """Detects trading strategies through pattern analysis."""

    # Thresholds for strategy detection
    EARLY_EXIT_THRESHOLD_HOURS = 24  # Exit within 24h considered early
    UNDERDOG_PRICE_THRESHOLD = 0.35  # Price below 35% considered underdog
    SCALP_HOLD_THRESHOLD_HOURS = 4  # Less than 4h considered scalp
    SWING_HOLD_THRESHOLD_HOURS = 72  # More than 72h considered swing

    def __init__(self) -> None:
        """Initialize the strategy detector."""
        self._signals: list[StrategySignal] = []

    def _detect_early_exit(
        self,
        processed_trades: list[ProcessedTrade]
    ) -> Optional[StrategySignal]:
        """Detect early exit strategy (selling before resolution).

        Args:
            processed_trades: List of processed trades.

        Returns:
            StrategySignal if pattern detected, None otherwise.
        """
        early_exits = []
        total_sells = 0

        for pt in processed_trades:
            if pt.trade.is_sell and pt.is_closing:
                total_sells += 1

                # Check if sold within threshold of buying
                # Look for corresponding buy
                for buy_pt in processed_trades:
                    if (buy_pt.trade.is_buy and
                        buy_pt.trade.market_id == pt.trade.market_id and
                        buy_pt.trade.outcome == pt.trade.outcome and
                        buy_pt.trade.timestamp < pt.trade.timestamp):

                        hold_time = (pt.trade.timestamp - buy_pt.trade.timestamp).total_seconds() / 3600

                        if hold_time < self.EARLY_EXIT_THRESHOLD_HOURS:
                            early_exits.append(pt.trade.id)
                        break

        if total_sells == 0:
            return None

        early_exit_rate = len(early_exits) / total_sells

        if early_exit_rate >= 0.4:  # 40% or more trades are early exits
            return StrategySignal(
                strategy=TradingStrategy.EARLY_EXIT,
                confidence=min(early_exit_rate, 1.0),
                evidence=[
                    f"{len(early_exits)}/{total_sells} trades closed within {self.EARLY_EXIT_THRESHOLD_HOURS}h",
                    f"Early exit rate: {early_exit_rate:.1%}"
                ],
                sample_trades=early_exits[:5]
            )

        return None

    def _detect_contrarian(
        self,
        processed_trades: list[ProcessedTrade]
    ) -> Optional[StrategySignal]:
        """Detect contrarian strategy (buying underdogs).

        Args:
            processed_trades: List of processed trades.

        Returns:
            StrategySignal if pattern detected, None otherwise.
        """
        underdog_buys = []
        total_buys = 0

        for pt in processed_trades:
            if pt.trade.is_buy:
                total_buys += 1

                # Check if buying at low price (underdog)
                price = float(pt.trade.price)
                if price < self.UNDERDOG_PRICE_THRESHOLD:
                    underdog_buys.append(pt.trade.id)

        if total_buys == 0:
            return None

        underdog_rate = len(underdog_buys) / total_buys

        if underdog_rate >= 0.5:  # 50% or more buys are underdogs
            return StrategySignal(
                strategy=TradingStrategy.CONTRARIAN,
                confidence=min(underdog_rate, 1.0),
                evidence=[
                    f"{len(underdog_buys)}/{total_buys} buys at price < {self.UNDERDOG_PRICE_THRESHOLD}",
                    f"Underdog buy rate: {underdog_rate:.1%}",
                    "Trader favors low-probability outcomes"
                ],
                sample_trades=underdog_buys[:5]
            )

        return None

    def _detect_momentum(
        self,
        processed_trades: list[ProcessedTrade]
    ) -> Optional[StrategySignal]:
        """Detect momentum strategy (buying favorites/following trends).

        Args:
            processed_trades: List of processed trades.

        Returns:
            StrategySignal if pattern detected, None otherwise.
        """
        favorite_buys = []
        total_buys = 0

        for pt in processed_trades:
            if pt.trade.is_buy:
                total_buys += 1

                # Check if buying at high price (favorite)
                price = float(pt.trade.price)
                if price > 0.65:  # Price above 65% considered favorite
                    favorite_buys.append(pt.trade.id)

        if total_buys == 0:
            return None

        favorite_rate = len(favorite_buys) / total_buys

        if favorite_rate >= 0.5:
            return StrategySignal(
                strategy=TradingStrategy.MOMENTUM,
                confidence=min(favorite_rate, 1.0),
                evidence=[
                    f"{len(favorite_buys)}/{total_buys} buys at price > 0.65",
                    f"Favorite buy rate: {favorite_rate:.1%}",
                    "Trader follows consensus/momentum"
                ],
                sample_trades=favorite_buys[:5]
            )

        return None

    def _detect_arbitrage(
        self,
        processed_trades: list[ProcessedTrade]
    ) -> Optional[StrategySignal]:
        """Detect arbitrage strategy (opposite positions in same market).

        Args:
            processed_trades: List of processed trades.

        Returns:
            StrategySignal if pattern detected, None otherwise.
        """
        # Group trades by market
        market_positions: dict[str, set[str]] = defaultdict(set)
        market_trades: dict[str, list[str]] = defaultdict(list)

        for pt in processed_trades:
            if pt.trade.is_buy:
                market_positions[pt.trade.market_id].add(pt.trade.outcome)
                market_trades[pt.trade.market_id].append(pt.trade.id)

        # Find markets with positions on multiple outcomes
        arbitrage_markets = []
        sample_trades = []

        for market_id, outcomes in market_positions.items():
            if len(outcomes) > 1:
                arbitrage_markets.append(market_id)
                sample_trades.extend(market_trades[market_id][:2])

        if not arbitrage_markets:
            return None

        total_markets = len(market_positions)
        arb_rate = len(arbitrage_markets) / total_markets if total_markets > 0 else 0

        if arb_rate >= 0.2:  # 20% or more markets have opposing positions
            return StrategySignal(
                strategy=TradingStrategy.ARBITRAGE,
                confidence=min(arb_rate * 2, 1.0),  # Scale up confidence
                evidence=[
                    f"{len(arbitrage_markets)}/{total_markets} markets with opposing positions",
                    f"Arbitrage rate: {arb_rate:.1%}",
                    "Trader takes positions on multiple outcomes"
                ],
                sample_trades=sample_trades[:5]
            )

        return None

    def _detect_scalping(
        self,
        processed_trades: list[ProcessedTrade]
    ) -> Optional[StrategySignal]:
        """Detect scalping strategy (quick small profits).

        Args:
            processed_trades: List of processed trades.

        Returns:
            StrategySignal if pattern detected, None otherwise.
        """
        scalp_trades = []
        closed_trades = 0

        for pt in processed_trades:
            if pt.is_closing:
                closed_trades += 1

                # Find corresponding buy to calculate hold time
                for buy_pt in processed_trades:
                    if (buy_pt.trade.is_buy and
                        buy_pt.trade.market_id == pt.trade.market_id and
                        buy_pt.trade.outcome == pt.trade.outcome and
                        buy_pt.trade.timestamp < pt.trade.timestamp):

                        hold_time = (pt.trade.timestamp - buy_pt.trade.timestamp).total_seconds() / 3600

                        if hold_time < self.SCALP_HOLD_THRESHOLD_HOURS:
                            scalp_trades.append(pt.trade.id)
                        break

        if closed_trades == 0:
            return None

        scalp_rate = len(scalp_trades) / closed_trades

        if scalp_rate >= 0.5:
            return StrategySignal(
                strategy=TradingStrategy.SCALPING,
                confidence=min(scalp_rate, 1.0),
                evidence=[
                    f"{len(scalp_trades)}/{closed_trades} trades held < {self.SCALP_HOLD_THRESHOLD_HOURS}h",
                    f"Scalp rate: {scalp_rate:.1%}",
                    "Trader prefers quick in-and-out trades"
                ],
                sample_trades=scalp_trades[:5]
            )

        return None

    def _detect_swing(
        self,
        processed_trades: list[ProcessedTrade]
    ) -> Optional[StrategySignal]:
        """Detect swing trading strategy (medium-term holds).

        Args:
            processed_trades: List of processed trades.

        Returns:
            StrategySignal if pattern detected, None otherwise.
        """
        swing_trades = []
        closed_trades = 0

        for pt in processed_trades:
            if pt.is_closing:
                closed_trades += 1

                for buy_pt in processed_trades:
                    if (buy_pt.trade.is_buy and
                        buy_pt.trade.market_id == pt.trade.market_id and
                        buy_pt.trade.outcome == pt.trade.outcome and
                        buy_pt.trade.timestamp < pt.trade.timestamp):

                        hold_time = (pt.trade.timestamp - buy_pt.trade.timestamp).total_seconds() / 3600

                        if hold_time > self.SWING_HOLD_THRESHOLD_HOURS:
                            swing_trades.append(pt.trade.id)
                        break

        if closed_trades == 0:
            return None

        swing_rate = len(swing_trades) / closed_trades

        if swing_rate >= 0.4:
            return StrategySignal(
                strategy=TradingStrategy.SWING,
                confidence=min(swing_rate, 1.0),
                evidence=[
                    f"{len(swing_trades)}/{closed_trades} trades held > {self.SWING_HOLD_THRESHOLD_HOURS}h",
                    f"Swing rate: {swing_rate:.1%}",
                    "Trader prefers longer holding periods"
                ],
                sample_trades=swing_trades[:5]
            )

        return None

    def _analyze_behavior_traits(
        self,
        processed_trades: list[ProcessedTrade]
    ) -> dict[str, Any]:
        """Analyze behavioral traits from trading patterns.

        Args:
            processed_trades: List of processed trades.

        Returns:
            Dictionary of behavioral traits.
        """
        if not processed_trades:
            return {}

        # Calculate various behavioral metrics
        trade_sizes = [float(pt.trade.size) for pt in processed_trades]
        trade_times = [pt.trade.timestamp.hour for pt in processed_trades]

        # Trading frequency
        if len(processed_trades) >= 2:
            timestamps = sorted(pt.trade.timestamp for pt in processed_trades)
            time_span = (timestamps[-1] - timestamps[0]).days + 1
            trades_per_day = len(processed_trades) / time_span if time_span > 0 else 0
        else:
            trades_per_day = 0

        # Position sizing consistency
        import numpy as np
        size_std = np.std(trade_sizes) if len(trade_sizes) > 1 else 0
        size_mean = np.mean(trade_sizes) if trade_sizes else 0
        sizing_consistency = 1 - (size_std / size_mean) if size_mean > 0 else 0

        # Active hours
        from collections import Counter
        hour_counts = Counter(trade_times)
        most_active_hours = [h for h, _ in hour_counts.most_common(3)]

        # Category preference
        category_counts = Counter(pt.trade.market_category for pt in processed_trades)
        preferred_categories = [cat for cat, _ in category_counts.most_common(3)]

        return {
            "trading_frequency": {
                "trades_per_day": round(trades_per_day, 2),
                "total_active_days": time_span if 'time_span' in locals() else 0
            },
            "position_sizing": {
                "avg_size": round(size_mean, 2),
                "consistency_score": round(max(0, min(1, sizing_consistency)), 2)
            },
            "timing": {
                "most_active_hours_utc": most_active_hours,
                "weekend_trading": any(
                    pt.trade.timestamp.weekday() >= 5 for pt in processed_trades
                )
            },
            "market_preference": {
                "preferred_categories": preferred_categories,
                "category_distribution": {
                    cat: count for cat, count in category_counts.most_common()
                }
            }
        }

    def _analyze_risk_profile(
        self,
        processed_trades: list[ProcessedTrade]
    ) -> dict[str, Any]:
        """Analyze risk profile from trading behavior.

        Args:
            processed_trades: List of processed trades.

        Returns:
            Dictionary of risk profile metrics.
        """
        if not processed_trades:
            return {}

        # Calculate risk metrics
        prices = [float(pt.trade.price) for pt in processed_trades if pt.trade.is_buy]
        pnls = [float(pt.realized_pnl) for pt in processed_trades if pt.is_closing]

        # Average buy price indicates risk preference
        avg_buy_price = sum(prices) / len(prices) if prices else 0.5

        # Win/loss distribution
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0

        # Risk/reward ratio
        risk_reward = avg_win / avg_loss if avg_loss > 0 else 0

        # Consecutive losses (streak analysis)
        max_loss_streak = 0
        current_streak = 0
        for pnl in pnls:
            if pnl < 0:
                current_streak += 1
                max_loss_streak = max(max_loss_streak, current_streak)
            else:
                current_streak = 0

        # Determine risk level
        if avg_buy_price < 0.3:
            risk_level = "aggressive"
        elif avg_buy_price < 0.5:
            risk_level = "moderate-high"
        elif avg_buy_price < 0.7:
            risk_level = "moderate"
        else:
            risk_level = "conservative"

        return {
            "risk_level": risk_level,
            "avg_entry_price": round(avg_buy_price, 3),
            "risk_reward_ratio": round(risk_reward, 2),
            "max_loss_streak": max_loss_streak,
            "avg_win_usd": round(avg_win, 2),
            "avg_loss_usd": round(avg_loss, 2),
            "largest_single_loss": round(min(pnls), 2) if pnls else 0,
            "largest_single_win": round(max(pnls), 2) if pnls else 0
        }

    def detect(self, processed_trades: list[ProcessedTrade]) -> StrategyProfile:
        """Detect trading strategies from processed trades.

        Args:
            processed_trades: List of processed trades.

        Returns:
            StrategyProfile with detected strategies and analysis.
        """
        if not processed_trades:
            return StrategyProfile(
                primary_strategy=TradingStrategy.UNKNOWN,
                secondary_strategies=[],
                signals=[],
                behavior_traits={},
                risk_profile={}
            )

        logger.info(f"Detecting strategies from {len(processed_trades)} trades")

        self._signals = []

        # Run all detection methods
        detectors = [
            self._detect_early_exit,
            self._detect_contrarian,
            self._detect_momentum,
            self._detect_arbitrage,
            self._detect_scalping,
            self._detect_swing
        ]

        for detector in detectors:
            signal = detector(processed_trades)
            if signal:
                self._signals.append(signal)

        # Sort signals by confidence
        self._signals.sort(key=lambda s: s.confidence, reverse=True)

        # Determine primary and secondary strategies
        if self._signals:
            primary = self._signals[0].strategy
            secondary = [s.strategy for s in self._signals[1:3]]
        else:
            primary = TradingStrategy.UNKNOWN
            secondary = []

        # Analyze behavior and risk
        behavior_traits = self._analyze_behavior_traits(processed_trades)
        risk_profile = self._analyze_risk_profile(processed_trades)

        profile = StrategyProfile(
            primary_strategy=primary,
            secondary_strategies=secondary,
            signals=self._signals,
            behavior_traits=behavior_traits,
            risk_profile=risk_profile
        )

        logger.info(
            f"Strategy detection complete: Primary={primary.value}, "
            f"Signals detected={len(self._signals)}"
        )

        return profile
