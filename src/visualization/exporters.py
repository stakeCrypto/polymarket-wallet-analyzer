"""Export utilities for various output formats."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..analysis.llm_formatter import LLMExport
from ..analysis.performance import PerformanceMetrics
from ..analysis.strategy_detector import StrategyProfile
from ..config import get_config, get_logger
from ..scraper.data_processor import ProcessedTrade
from ..scraper.wallet_fetcher import Trade

logger = get_logger(__name__)


class BaseExporter:
    """Base class for exporters."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        """Initialize the exporter.

        Args:
            output_dir: Directory to save exports.
        """
        self.config = get_config()
        self.output_dir = output_dir or self.config.export.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_filename(self, wallet_address: str, extension: str) -> Path:
        """Generate timestamped filename.

        Args:
            wallet_address: Wallet address for filename.
            extension: File extension.

        Returns:
            Path to output file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_addr = wallet_address[:10]
        return self.output_dir / f"wallet_{short_addr}_{timestamp}.{extension}"


class JSONExporter(BaseExporter):
    """Export data to JSON format."""

    def export_llm_format(
        self,
        llm_export: LLMExport,
        wallet_address: str,
        filename: Optional[str] = None
    ) -> Path:
        """Export LLM-formatted data to JSON.

        Args:
            llm_export: LLMExport object.
            wallet_address: Wallet address.
            filename: Optional custom filename.

        Returns:
            Path to exported file.
        """
        output_path = Path(filename) if filename else self._generate_filename(wallet_address, "json")

        with open(output_path, "w") as f:
            f.write(llm_export.to_json(indent=2))

        logger.info(f"Exported LLM format to {output_path}")
        return output_path

    def export_trades(
        self,
        trades: list[Trade],
        wallet_address: str,
        filename: Optional[str] = None
    ) -> Path:
        """Export raw trades to JSON.

        Args:
            trades: List of Trade objects.
            wallet_address: Wallet address.
            filename: Optional custom filename.

        Returns:
            Path to exported file.
        """
        output_path = Path(filename) if filename else self._generate_filename(wallet_address, "trades.json")

        data = {
            "wallet_address": wallet_address,
            "trade_count": len(trades),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "trades": [t.to_dict() for t in trades]
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Exported {len(trades)} trades to {output_path}")
        return output_path

    def export_analysis(
        self,
        wallet_address: str,
        metrics: PerformanceMetrics,
        strategy: StrategyProfile,
        filename: Optional[str] = None
    ) -> Path:
        """Export analysis results to JSON.

        Args:
            wallet_address: Wallet address.
            metrics: Performance metrics.
            strategy: Strategy profile.
            filename: Optional custom filename.

        Returns:
            Path to exported file.
        """
        output_path = Path(filename) if filename else self._generate_filename(wallet_address, "analysis.json")

        data = {
            "wallet_address": wallet_address,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "performance_metrics": metrics.to_dict(),
            "strategy_profile": strategy.to_dict()
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Exported analysis to {output_path}")
        return output_path


class CSVExporter(BaseExporter):
    """Export data to CSV format."""

    def export_trades(
        self,
        trades: list[Trade],
        wallet_address: str,
        filename: Optional[str] = None
    ) -> Path:
        """Export trades to CSV.

        Args:
            trades: List of Trade objects.
            wallet_address: Wallet address.
            filename: Optional custom filename.

        Returns:
            Path to exported file.
        """
        output_path = Path(filename) if filename else self._generate_filename(wallet_address, "csv")

        if not trades:
            logger.warning("No trades to export")
            return output_path

        fieldnames = [
            "id", "wallet_address", "market_id", "market_title", "market_category",
            "outcome", "side", "price", "size", "cost_usd", "timestamp",
            "transaction_hash", "fee_usd"
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for trade in trades:
                writer.writerow(trade.to_dict())

        logger.info(f"Exported {len(trades)} trades to {output_path}")
        return output_path

    def export_processed_trades(
        self,
        processed_trades: list[ProcessedTrade],
        wallet_address: str,
        filename: Optional[str] = None
    ) -> Path:
        """Export processed trades with P&L to CSV.

        Args:
            processed_trades: List of ProcessedTrade objects.
            wallet_address: Wallet address.
            filename: Optional custom filename.

        Returns:
            Path to exported file.
        """
        output_path = Path(filename) if filename else self._generate_filename(wallet_address, "processed.csv")

        if not processed_trades:
            logger.warning("No processed trades to export")
            return output_path

        fieldnames = [
            "id", "wallet_address", "market_id", "market_title", "market_category",
            "outcome", "side", "price", "size", "cost_usd", "timestamp",
            "realized_pnl", "is_closing", "position_size_after", "cumulative_pnl"
        ]

        cumulative_pnl = 0.0

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for pt in processed_trades:
                cumulative_pnl += float(pt.realized_pnl)
                row = {
                    **pt.trade.to_dict(),
                    "realized_pnl": str(pt.realized_pnl),
                    "is_closing": pt.is_closing,
                    "position_size_after": str(pt.position_size_after),
                    "cumulative_pnl": str(cumulative_pnl)
                }
                writer.writerow(row)

        logger.info(f"Exported {len(processed_trades)} processed trades to {output_path}")
        return output_path

    def export_summary(
        self,
        wallet_address: str,
        metrics: PerformanceMetrics,
        strategy: StrategyProfile,
        filename: Optional[str] = None
    ) -> Path:
        """Export summary metrics to CSV.

        Args:
            wallet_address: Wallet address.
            metrics: Performance metrics.
            strategy: Strategy profile.
            filename: Optional custom filename.

        Returns:
            Path to exported file.
        """
        output_path = Path(filename) if filename else self._generate_filename(wallet_address, "summary.csv")

        rows = [
            ["Metric", "Value"],
            ["Wallet Address", wallet_address],
            ["Total Trades", metrics.total_trades],
            ["Total Volume (USD)", f"{metrics.total_volume_usd:.2f}"],
            ["Total P&L (USD)", f"{metrics.total_realized_pnl:.2f}"],
            ["Win Rate", f"{metrics.win_rate:.2%}"],
            ["Profit Factor", f"{metrics.profit_factor:.2f}"],
            ["Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}"],
            ["Max Drawdown", f"{metrics.max_drawdown:.2f}"],
            ["ROI", f"{metrics.roi_percent:.2f}%"],
            ["Best Trade P&L", f"{metrics.best_trade_pnl:.2f}"],
            ["Worst Trade P&L", f"{metrics.worst_trade_pnl:.2f}"],
            ["Avg Hold Time (hours)", f"{metrics.avg_hold_time_hours:.2f}"],
            ["Primary Strategy", strategy.primary_strategy.value],
            ["Risk Level", strategy.risk_profile.get("risk_level", "unknown")],
            ["Active Days", metrics.active_days],
            ["First Trade", metrics.first_trade_date.isoformat() if metrics.first_trade_date else "N/A"],
            ["Last Trade", metrics.last_trade_date.isoformat() if metrics.last_trade_date else "N/A"],
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        logger.info(f"Exported summary to {output_path}")
        return output_path


class HTMLExporter(BaseExporter):
    """Export data to HTML report format."""

    HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Polymarket Wallet Analysis - {wallet_short}</title>
    <style>
        :root {{
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --text-primary: #eaeaea;
            --text-secondary: #a0a0a0;
            --accent: #2196F3;
            --profit: #00C853;
            --loss: #FF1744;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        h1 {{
            color: var(--accent);
            margin-bottom: 0.5rem;
        }}

        h2 {{
            color: var(--text-primary);
            margin: 2rem 0 1rem;
            border-bottom: 2px solid var(--accent);
            padding-bottom: 0.5rem;
        }}

        .subtitle {{
            color: var(--text-secondary);
            margin-bottom: 2rem;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .card {{
            background: var(--bg-secondary);
            border-radius: 8px;
            padding: 1.5rem;
        }}

        .card-title {{
            color: var(--text-secondary);
            font-size: 0.875rem;
            margin-bottom: 0.5rem;
        }}

        .card-value {{
            font-size: 1.5rem;
            font-weight: bold;
        }}

        .card-value.profit {{
            color: var(--profit);
        }}

        .card-value.loss {{
            color: var(--loss);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--bg-secondary);
            border-radius: 8px;
            overflow: hidden;
        }}

        th, td {{
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid var(--bg-primary);
        }}

        th {{
            background: var(--bg-primary);
            color: var(--accent);
            font-weight: 600;
        }}

        tr:hover {{
            background: rgba(33, 150, 243, 0.1);
        }}

        .insight {{
            background: var(--bg-secondary);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
            border-left: 4px solid var(--accent);
        }}

        .footer {{
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid var(--bg-secondary);
            color: var(--text-secondary);
            font-size: 0.875rem;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Polymarket Wallet Analysis</h1>
        <p class="subtitle">Wallet: {wallet_address}</p>

        <div class="grid">
            <div class="card">
                <div class="card-title">Total Trades</div>
                <div class="card-value">{total_trades}</div>
            </div>
            <div class="card">
                <div class="card-title">Total Volume</div>
                <div class="card-value">${total_volume:,.0f}</div>
            </div>
            <div class="card">
                <div class="card-title">Net P&L</div>
                <div class="card-value {pnl_class}">${net_pnl:+,.0f}</div>
            </div>
            <div class="card">
                <div class="card-title">Win Rate</div>
                <div class="card-value">{win_rate:.1%}</div>
            </div>
            <div class="card">
                <div class="card-title">ROI</div>
                <div class="card-value {roi_class}">{roi:+.1f}%</div>
            </div>
            <div class="card">
                <div class="card-title">Primary Strategy</div>
                <div class="card-value">{strategy}</div>
            </div>
        </div>

        <h2>Key Insights</h2>
        {insights_html}

        <h2>Top Performing Trades</h2>
        <table>
            <thead>
                <tr>
                    <th>Market</th>
                    <th>P&L</th>
                    <th>ROI</th>
                    <th>Hold Time</th>
                </tr>
            </thead>
            <tbody>
                {top_trades_html}
            </tbody>
        </table>

        <h2>Performance by Category</h2>
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Volume</th>
                    <th>P&L</th>
                    <th>Win Rate</th>
                </tr>
            </thead>
            <tbody>
                {category_html}
            </tbody>
        </table>

        <div class="footer">
            Generated by Polymarket Wallet Analyzer v1.0.0 | {generated_at}
        </div>
    </div>
</body>
</html>
"""

    def export(
        self,
        llm_export: LLMExport,
        wallet_address: str,
        filename: Optional[str] = None
    ) -> Path:
        """Export analysis to HTML report.

        Args:
            llm_export: LLMExport object with analysis data.
            wallet_address: Wallet address.
            filename: Optional custom filename.

        Returns:
            Path to exported file.
        """
        output_path = Path(filename) if filename else self._generate_filename(wallet_address, "html")

        # Prepare template variables
        summary = llm_export.wallet_summary
        metrics = llm_export.performance_metrics

        pnl_class = "profit" if summary["net_pnl_usd"] >= 0 else "loss"
        roi_class = "profit" if metrics["roi_percent"] >= 0 else "loss"

        # Build insights HTML
        insights_html = "\n".join(
            f'<div class="insight">{insight}</div>'
            for insight in llm_export.key_insights
        )

        # Build top trades HTML
        top_trades_rows = []
        for trade in llm_export.top_trades[:5]:
            pnl_class_trade = "profit" if trade["pnl_usd"] >= 0 else "loss"
            hold_time = f"{trade['hold_time_hours']:.1f}h" if trade.get("hold_time_hours") else "N/A"
            top_trades_rows.append(f"""
                <tr>
                    <td>{trade['market'][:50]}...</td>
                    <td class="{pnl_class_trade}">${trade['pnl_usd']:+,.0f}</td>
                    <td>{trade['pnl_percent']:+.1f}%</td>
                    <td>{hold_time}</td>
                </tr>
            """)
        top_trades_html = "\n".join(top_trades_rows)

        # Build category HTML
        category_rows = []
        for cat, data in metrics.get("by_category", {}).items():
            pnl_class_cat = "profit" if data["pnl"] >= 0 else "loss"
            category_rows.append(f"""
                <tr>
                    <td>{cat.title()}</td>
                    <td>${data['volume']:,.0f}</td>
                    <td class="{pnl_class_cat}">${data['pnl']:+,.0f}</td>
                    <td>{data['win_rate']:.1%}</td>
                </tr>
            """)
        category_html = "\n".join(category_rows)

        # Render template
        html_content = self.HTML_TEMPLATE.format(
            wallet_short=f"{wallet_address[:6]}...{wallet_address[-4:]}",
            wallet_address=wallet_address,
            total_trades=summary["total_trades"],
            total_volume=summary["total_volume_usd"],
            net_pnl=summary["net_pnl_usd"],
            pnl_class=pnl_class,
            win_rate=summary["win_rate"],
            roi=metrics["roi_percent"],
            roi_class=roi_class,
            strategy=llm_export.detected_strategy["primary_strategy"].replace("_", " ").title(),
            insights_html=insights_html,
            top_trades_html=top_trades_html,
            category_html=category_html,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"Exported HTML report to {output_path}")
        return output_path
