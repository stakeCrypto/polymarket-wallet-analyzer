"""LLM-optimized export formatter for trading analysis."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from ..config import get_logger
from ..scraper.data_processor import ProcessedTrade
from ..scraper.wallet_fetcher import Position, Trade
from .performance import PerformanceAnalyzer, PerformanceMetrics, TradePerformance
from .strategy_detector import StrategyDetector, StrategyProfile

logger = get_logger(__name__)


@dataclass
class LLMExport:
    """Structured export optimized for LLM consumption."""

    wallet_summary: dict[str, Any]
    performance_metrics: dict[str, Any]
    top_trades: list[dict[str, Any]]
    worst_trades: list[dict[str, Any]]
    detected_strategy: dict[str, Any]
    risk_profile: dict[str, Any]
    key_insights: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "wallet_summary": self.wallet_summary,
            "performance_metrics": self.performance_metrics,
            "top_trades": self.top_trades,
            "worst_trades": self.worst_trades,
            "detected_strategy": self.detected_strategy,
            "risk_profile": self.risk_profile,
            "key_insights": self.key_insights,
            "metadata": self.metadata
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)


class LLMFormatter:
    """Formats trading analysis for LLM consumption."""

    def __init__(self) -> None:
        """Initialize the LLM formatter."""
        self.performance_analyzer = PerformanceAnalyzer()
        self.strategy_detector = StrategyDetector()

    def _generate_wallet_summary(
        self,
        wallet_address: str,
        trades: list[Trade],
        positions: list[Position],
        metrics: PerformanceMetrics
    ) -> dict[str, Any]:
        """Generate wallet summary section.

        Args:
            wallet_address: Wallet address.
            trades: List of trades.
            positions: List of current positions.
            metrics: Performance metrics.

        Returns:
            Wallet summary dictionary.
        """
        return {
            "address": wallet_address,
            "total_trades": metrics.total_trades,
            "total_volume_usd": round(metrics.total_volume_usd, 2),
            "net_pnl_usd": round(metrics.total_realized_pnl, 2),
            "win_rate": round(metrics.win_rate, 3),
            "active_positions": len(positions),
            "active_since": metrics.first_trade_date.strftime("%Y-%m-%d") if metrics.first_trade_date else None,
            "last_activity": metrics.last_trade_date.strftime("%Y-%m-%d") if metrics.last_trade_date else None,
            "active_days": metrics.active_days
        }

    def _generate_key_insights(
        self,
        metrics: PerformanceMetrics,
        strategy: StrategyProfile
    ) -> list[str]:
        """Generate key insights from analysis.

        Args:
            metrics: Performance metrics.
            strategy: Strategy profile.

        Returns:
            List of insight strings.
        """
        insights = []

        # Performance insights
        if metrics.win_rate >= 0.6:
            insights.append(
                f"High win rate of {metrics.win_rate:.1%} indicates strong market timing ability"
            )
        elif metrics.win_rate <= 0.4:
            insights.append(
                f"Win rate of {metrics.win_rate:.1%} suggests room for improvement in trade selection"
            )

        if metrics.profit_factor > 2:
            insights.append(
                f"Excellent profit factor of {metrics.profit_factor:.2f} (gross profit/loss ratio)"
            )
        elif metrics.profit_factor < 1:
            insights.append(
                f"Profit factor below 1.0 ({metrics.profit_factor:.2f}) indicates losses exceed gains"
            )

        # ROI insights
        if metrics.roi_percent > 20:
            insights.append(
                f"Strong ROI of {metrics.roi_percent:.1f}% on invested capital"
            )
        elif metrics.roi_percent < 0:
            insights.append(
                f"Negative ROI of {metrics.roi_percent:.1f}% - strategy review recommended"
            )

        # Strategy insights
        primary = strategy.primary_strategy.value
        if primary != "unknown":
            insights.append(
                f"Primary trading strategy detected: {primary.replace('_', ' ').title()}"
            )

        # Risk insights
        risk_level = strategy.risk_profile.get("risk_level", "unknown")
        if risk_level == "aggressive":
            insights.append(
                "Aggressive risk profile - focuses on high-risk/high-reward positions"
            )
        elif risk_level == "conservative":
            insights.append(
                "Conservative risk profile - prefers higher-probability outcomes"
            )

        # Behavioral insights
        behavior = strategy.behavior_traits
        if behavior.get("trading_frequency", {}).get("trades_per_day", 0) > 5:
            insights.append(
                "High-frequency trader with multiple daily trades"
            )

        sizing = behavior.get("position_sizing", {})
        if sizing.get("consistency_score", 0) > 0.8:
            insights.append(
                "Consistent position sizing indicates disciplined risk management"
            )
        elif sizing.get("consistency_score", 0) < 0.4:
            insights.append(
                "Inconsistent position sizing may indicate emotional trading"
            )

        # Category preference
        prefs = behavior.get("market_preference", {}).get("preferred_categories", [])
        if prefs:
            insights.append(
                f"Market focus: primarily trades {', '.join(prefs[:2])} markets"
            )

        # Max drawdown warning
        if metrics.max_drawdown_percent > 0.3:
            insights.append(
                f"Warning: Maximum drawdown of {metrics.max_drawdown_percent:.1%} indicates significant risk exposure"
            )

        return insights

    def _format_trade_for_llm(
        self,
        trade_perf: TradePerformance
    ) -> dict[str, Any]:
        """Format a trade performance record for LLM consumption.

        Args:
            trade_perf: Trade performance object.

        Returns:
            Formatted trade dictionary.
        """
        return {
            "market": trade_perf.market_title[:100],  # Truncate long titles
            "category": trade_perf.market_category,
            "entry_price": round(trade_perf.entry_price, 3),
            "exit_price": round(trade_perf.exit_price, 3) if trade_perf.exit_price else None,
            "position_size": round(trade_perf.size, 2),
            "pnl_usd": round(trade_perf.pnl, 2),
            "pnl_percent": round(trade_perf.pnl_percent, 1),
            "hold_time_hours": round(trade_perf.hold_time_hours, 1) if trade_perf.hold_time_hours else None,
            "entry_date": trade_perf.entry_time.strftime("%Y-%m-%d %H:%M") if trade_perf.entry_time else None,
            "exit_date": trade_perf.exit_time.strftime("%Y-%m-%d %H:%M") if trade_perf.exit_time else None
        }

    def format(
        self,
        wallet_address: str,
        trades: list[Trade],
        positions: list[Position],
        processed_trades: list[ProcessedTrade]
    ) -> LLMExport:
        """Generate LLM-optimized export from trading data.

        Args:
            wallet_address: Wallet address.
            trades: List of raw trades.
            positions: List of current positions.
            processed_trades: List of processed trades with P&L.

        Returns:
            LLMExport object with formatted analysis.
        """
        logger.info(f"Formatting LLM export for wallet {wallet_address}")

        # Run analysis
        metrics = self.performance_analyzer.analyze(processed_trades)
        strategy = self.strategy_detector.detect(processed_trades)

        # Get top and worst trades
        top_trades = self.performance_analyzer.get_top_trades(n=10)
        worst_trades = self.performance_analyzer.get_worst_trades(n=5)

        # Generate export
        export = LLMExport(
            wallet_summary=self._generate_wallet_summary(
                wallet_address, trades, positions, metrics
            ),
            performance_metrics=metrics.to_dict(),
            top_trades=[self._format_trade_for_llm(t) for t in top_trades],
            worst_trades=[self._format_trade_for_llm(t) for t in worst_trades],
            detected_strategy=strategy.to_dict(),
            risk_profile=strategy.risk_profile,
            key_insights=self._generate_key_insights(metrics, strategy),
            metadata={
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "analyzer_version": "1.0.0",
                "total_trades_analyzed": len(processed_trades),
                "data_quality": {
                    "trades_parsed": len(trades),
                    "trades_processed": len(processed_trades),
                    "positions_active": len(positions)
                }
            }
        )

        logger.info("LLM export formatting complete")

        return export

    def format_compact(
        self,
        wallet_address: str,
        trades: list[Trade],
        positions: list[Position],
        processed_trades: list[ProcessedTrade]
    ) -> dict[str, Any]:
        """Generate compact summary for quick LLM analysis.

        Args:
            wallet_address: Wallet address.
            trades: List of raw trades.
            positions: List of current positions.
            processed_trades: List of processed trades.

        Returns:
            Compact summary dictionary.
        """
        full_export = self.format(
            wallet_address, trades, positions, processed_trades
        )

        return {
            "wallet": wallet_address[:10] + "...",
            "summary": {
                "trades": full_export.wallet_summary["total_trades"],
                "volume": f"${full_export.wallet_summary['total_volume_usd']:,.0f}",
                "pnl": f"${full_export.wallet_summary['net_pnl_usd']:,.0f}",
                "win_rate": f"{full_export.wallet_summary['win_rate']:.1%}",
                "roi": f"{full_export.performance_metrics['roi_percent']:.1f}%"
            },
            "strategy": full_export.detected_strategy["primary_strategy"],
            "risk_level": full_export.risk_profile.get("risk_level", "unknown"),
            "top_insight": full_export.key_insights[0] if full_export.key_insights else "No insights available",
            "best_trade_pnl": f"${full_export.top_trades[0]['pnl_usd']:,.0f}" if full_export.top_trades else "N/A",
            "worst_trade_pnl": f"${full_export.worst_trades[0]['pnl_usd']:,.0f}" if full_export.worst_trades else "N/A"
        }
