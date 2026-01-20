"""Interactive Plotly dashboard for trading analysis visualization."""

import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..analysis.performance import PerformanceMetrics
from ..analysis.strategy_detector import StrategyProfile
from ..config import get_config, get_logger
from ..scraper.data_processor import ProcessedTrade

logger = get_logger(__name__)


class Dashboard:
    """Interactive Plotly dashboard generator."""

    # Color scheme
    COLORS = {
        "profit": "#00C853",
        "loss": "#FF1744",
        "neutral": "#2196F3",
        "background": "#1a1a2e",
        "text": "#eaeaea",
        "grid": "#333355"
    }

    # Category colors
    CATEGORY_COLORS = {
        "politics": "#FF6B6B",
        "crypto": "#4ECDC4",
        "sports": "#45B7D1",
        "finance": "#96CEB4",
        "entertainment": "#FFEAA7",
        "science": "#DDA0DD",
        "world": "#98D8C8",
        "other": "#C9C9C9"
    }

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        """Initialize the dashboard generator.

        Args:
            output_dir: Directory to save dashboard files.
        """
        self.config = get_config()
        self.output_dir = output_dir or self.config.export.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _create_pnl_timeline(
        self,
        df: pd.DataFrame
    ) -> go.Figure:
        """Create cumulative P&L timeline chart.

        Args:
            df: DataFrame with processed trade data.

        Returns:
            Plotly figure.
        """
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig

        fig = go.Figure()

        # Cumulative P&L line
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["cumulative_pnl"],
            mode="lines+markers",
            name="Cumulative P&L",
            line=dict(color=self.COLORS["neutral"], width=2),
            marker=dict(size=4),
            fill="tozeroy",
            fillcolor="rgba(33, 150, 243, 0.1)"
        ))

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color=self.COLORS["text"], opacity=0.5)

        # Color regions
        max_pnl = df["cumulative_pnl"].max()
        min_pnl = df["cumulative_pnl"].min()

        fig.update_layout(
            title="Cumulative P&L Over Time",
            xaxis_title="Date",
            yaxis_title="P&L (USD)",
            template="plotly_dark",
            paper_bgcolor=self.COLORS["background"],
            plot_bgcolor=self.COLORS["background"],
            font=dict(color=self.COLORS["text"]),
            hovermode="x unified"
        )

        return fig

    def _create_win_rate_by_category(
        self,
        category_metrics: dict[str, dict[str, float]]
    ) -> go.Figure:
        """Create win rate by category bar chart.

        Args:
            category_metrics: Dictionary of category metrics.

        Returns:
            Plotly figure.
        """
        if not category_metrics:
            fig = go.Figure()
            fig.add_annotation(text="No category data", xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig

        categories = list(category_metrics.keys())
        win_rates = [category_metrics[c]["win_rate"] * 100 for c in categories]
        trade_counts = [category_metrics[c]["trade_count"] for c in categories]
        colors = [self.CATEGORY_COLORS.get(c, self.CATEGORY_COLORS["other"]) for c in categories]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=categories,
            y=win_rates,
            marker_color=colors,
            text=[f"{wr:.1f}%<br>({tc} trades)" for wr, tc in zip(win_rates, trade_counts)],
            textposition="auto",
            name="Win Rate"
        ))

        # Add 50% reference line
        fig.add_hline(y=50, line_dash="dash", line_color=self.COLORS["text"],
                     opacity=0.5, annotation_text="50%")

        fig.update_layout(
            title="Win Rate by Market Category",
            xaxis_title="Category",
            yaxis_title="Win Rate (%)",
            template="plotly_dark",
            paper_bgcolor=self.COLORS["background"],
            plot_bgcolor=self.COLORS["background"],
            font=dict(color=self.COLORS["text"]),
            yaxis=dict(range=[0, 100])
        )

        return fig

    def _create_position_sizing_chart(
        self,
        df: pd.DataFrame
    ) -> go.Figure:
        """Create position sizing over time chart.

        Args:
            df: DataFrame with trade data.

        Returns:
            Plotly figure.
        """
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig

        # Resample to daily for cleaner visualization
        df_daily = df.set_index("timestamp").resample("D").agg({
            "cost_usd": "sum",
            "realized_pnl": "sum"
        }).reset_index()

        colors = [
            self.COLORS["profit"] if pnl >= 0 else self.COLORS["loss"]
            for pnl in df_daily["realized_pnl"]
        ]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df_daily["timestamp"],
            y=df_daily["cost_usd"],
            marker_color=colors,
            name="Daily Volume",
            opacity=0.8
        ))

        fig.update_layout(
            title="Daily Trading Volume (colored by P&L)",
            xaxis_title="Date",
            yaxis_title="Volume (USD)",
            template="plotly_dark",
            paper_bgcolor=self.COLORS["background"],
            plot_bgcolor=self.COLORS["background"],
            font=dict(color=self.COLORS["text"])
        )

        return fig

    def _create_top_markets_chart(
        self,
        df: pd.DataFrame,
        n: int = 10
    ) -> go.Figure:
        """Create top traded markets chart.

        Args:
            df: DataFrame with trade data.
            n: Number of top markets to show.

        Returns:
            Plotly figure.
        """
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig

        # Aggregate by market
        market_stats = df.groupby("market_title").agg({
            "cost_usd": "sum",
            "realized_pnl": "sum",
            "id": "count"
        }).rename(columns={"id": "trade_count"}).reset_index()

        # Get top N by volume
        top_markets = market_stats.nlargest(n, "cost_usd")

        # Truncate long titles
        top_markets["market_short"] = top_markets["market_title"].str[:40] + "..."

        colors = [
            self.COLORS["profit"] if pnl >= 0 else self.COLORS["loss"]
            for pnl in top_markets["realized_pnl"]
        ]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=top_markets["market_short"],
            x=top_markets["cost_usd"],
            orientation="h",
            marker_color=colors,
            text=[f"${v:,.0f} ({tc} trades, P&L: ${pnl:+,.0f})"
                  for v, tc, pnl in zip(top_markets["cost_usd"],
                                       top_markets["trade_count"],
                                       top_markets["realized_pnl"])],
            textposition="auto",
            name="Volume"
        ))

        fig.update_layout(
            title=f"Top {n} Markets by Volume",
            xaxis_title="Total Volume (USD)",
            yaxis_title="",
            template="plotly_dark",
            paper_bgcolor=self.COLORS["background"],
            plot_bgcolor=self.COLORS["background"],
            font=dict(color=self.COLORS["text"]),
            height=400 + n * 25,
            yaxis=dict(autorange="reversed")
        )

        return fig

    def _create_strategy_radar(
        self,
        strategy: StrategyProfile
    ) -> go.Figure:
        """Create strategy profile radar chart.

        Args:
            strategy: Strategy profile.

        Returns:
            Plotly figure.
        """
        # Extract signal confidences
        strategy_scores = {
            "Early Exit": 0,
            "Contrarian": 0,
            "Momentum": 0,
            "Arbitrage": 0,
            "Scalping": 0,
            "Swing": 0
        }

        for signal in strategy.signals:
            strategy_name = signal.strategy.value.replace("_", " ").title()
            if strategy_name in strategy_scores:
                strategy_scores[strategy_name] = signal.confidence

        categories = list(strategy_scores.keys())
        values = list(strategy_scores.values())
        values.append(values[0])  # Close the polygon

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(33, 150, 243, 0.3)",
            line=dict(color=self.COLORS["neutral"], width=2),
            name="Strategy Profile"
        ))

        fig.update_layout(
            title="Strategy Profile",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickformat=".0%"
                ),
                bgcolor=self.COLORS["background"]
            ),
            template="plotly_dark",
            paper_bgcolor=self.COLORS["background"],
            font=dict(color=self.COLORS["text"]),
            showlegend=False
        )

        return fig

    def _create_pnl_distribution(
        self,
        df: pd.DataFrame
    ) -> go.Figure:
        """Create P&L distribution histogram.

        Args:
            df: DataFrame with trade data.

        Returns:
            Plotly figure.
        """
        if df.empty or "realized_pnl" not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="No P&L data", xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig

        pnl_data = df[df["is_closing"]]["realized_pnl"]

        if pnl_data.empty:
            fig = go.Figure()
            fig.add_annotation(text="No closed trades", xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig

        fig = go.Figure()

        # Split into profit and loss for coloring
        profits = pnl_data[pnl_data >= 0]
        losses = pnl_data[pnl_data < 0]

        fig.add_trace(go.Histogram(
            x=profits,
            name="Profits",
            marker_color=self.COLORS["profit"],
            opacity=0.7,
            nbinsx=30
        ))

        fig.add_trace(go.Histogram(
            x=losses,
            name="Losses",
            marker_color=self.COLORS["loss"],
            opacity=0.7,
            nbinsx=30
        ))

        fig.add_vline(x=0, line_dash="dash", line_color=self.COLORS["text"])

        fig.update_layout(
            title="P&L Distribution",
            xaxis_title="P&L (USD)",
            yaxis_title="Frequency",
            template="plotly_dark",
            paper_bgcolor=self.COLORS["background"],
            plot_bgcolor=self.COLORS["background"],
            font=dict(color=self.COLORS["text"]),
            barmode="overlay"
        )

        return fig

    def generate(
        self,
        wallet_address: str,
        df: pd.DataFrame,
        metrics: PerformanceMetrics,
        strategy: StrategyProfile
    ) -> Path:
        """Generate complete interactive dashboard.

        Args:
            wallet_address: Wallet address.
            df: DataFrame with processed trade data.
            metrics: Performance metrics.
            strategy: Strategy profile.

        Returns:
            Path to generated HTML file.
        """
        logger.info(f"Generating dashboard for wallet {wallet_address}")

        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Cumulative P&L Over Time",
                "Win Rate by Category",
                "Daily Trading Volume",
                "Top Markets by Volume",
                "P&L Distribution",
                "Strategy Profile"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "polar"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )

        # Add individual charts (simplified for subplot compatibility)
        # 1. P&L Timeline
        if not df.empty:
            fig.add_trace(go.Scatter(
                x=df["timestamp"],
                y=df["cumulative_pnl"],
                mode="lines",
                name="Cumulative P&L",
                line=dict(color=self.COLORS["neutral"]),
                fill="tozeroy"
            ), row=1, col=1)

        # 2. Win Rate by Category
        if metrics.category_metrics:
            categories = list(metrics.category_metrics.keys())
            win_rates = [metrics.category_metrics[c]["win_rate"] * 100 for c in categories]
            colors = [self.CATEGORY_COLORS.get(c, self.CATEGORY_COLORS["other"]) for c in categories]

            fig.add_trace(go.Bar(
                x=categories,
                y=win_rates,
                marker_color=colors,
                name="Win Rate"
            ), row=1, col=2)

        # 3. Position sizing
        if not df.empty:
            df_daily = df.set_index("timestamp").resample("D").agg({
                "cost_usd": "sum"
            }).reset_index()

            fig.add_trace(go.Bar(
                x=df_daily["timestamp"],
                y=df_daily["cost_usd"],
                marker_color=self.COLORS["neutral"],
                name="Volume",
                opacity=0.8
            ), row=2, col=1)

        # 4. Top markets
        if not df.empty:
            market_stats = df.groupby("market_title").agg({
                "cost_usd": "sum"
            }).reset_index().nlargest(8, "cost_usd")
            market_stats["market_short"] = market_stats["market_title"].str[:30] + "..."

            fig.add_trace(go.Bar(
                y=market_stats["market_short"],
                x=market_stats["cost_usd"],
                orientation="h",
                marker_color=self.COLORS["neutral"],
                name="Market Volume"
            ), row=2, col=2)

        # 5. P&L Distribution
        if not df.empty and "realized_pnl" in df.columns:
            pnl_data = df[df["is_closing"]]["realized_pnl"]
            fig.add_trace(go.Histogram(
                x=pnl_data,
                marker_color=self.COLORS["neutral"],
                name="P&L Distribution",
                nbinsx=30
            ), row=3, col=1)

        # 6. Strategy Radar
        strategy_scores = {"Early Exit": 0, "Contrarian": 0, "Momentum": 0,
                         "Arbitrage": 0, "Scalping": 0, "Swing": 0}
        for signal in strategy.signals:
            name = signal.strategy.value.replace("_", " ").title()
            if name in strategy_scores:
                strategy_scores[name] = signal.confidence

        categories = list(strategy_scores.keys())
        values = list(strategy_scores.values()) + [list(strategy_scores.values())[0]]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(33, 150, 243, 0.3)",
            line=dict(color=self.COLORS["neutral"]),
            name="Strategy"
        ), row=3, col=2)

        # Update layout
        short_address = f"{wallet_address[:6]}...{wallet_address[-4:]}"
        fig.update_layout(
            title=dict(
                text=f"Polymarket Wallet Analysis: {short_address}",
                font=dict(size=20)
            ),
            template="plotly_dark",
            paper_bgcolor=self.COLORS["background"],
            plot_bgcolor=self.COLORS["background"],
            font=dict(color=self.COLORS["text"]),
            height=1200,
            showlegend=False
        )

        # Add summary text
        summary_text = (
            f"<b>Summary</b><br>"
            f"Total Trades: {metrics.total_trades}<br>"
            f"Total Volume: ${metrics.total_volume_usd:,.0f}<br>"
            f"Net P&L: ${metrics.total_realized_pnl:,.0f}<br>"
            f"Win Rate: {metrics.win_rate:.1%}<br>"
            f"ROI: {metrics.roi_percent:.1f}%<br>"
            f"Strategy: {strategy.primary_strategy.value.replace('_', ' ').title()}"
        )

        fig.add_annotation(
            text=summary_text,
            xref="paper", yref="paper",
            x=1.02, y=1,
            showarrow=False,
            font=dict(size=12),
            align="left",
            bgcolor="rgba(0,0,0,0.5)",
            borderpad=10
        )

        # Save to HTML
        output_path = self.output_dir / f"dashboard_{wallet_address[:10]}.html"
        fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)

        logger.info(f"Dashboard saved to {output_path}")

        return output_path

    def open_dashboard(self, dashboard_path: Path) -> None:
        """Open dashboard in default web browser.

        Args:
            dashboard_path: Path to HTML dashboard file.
        """
        webbrowser.open(f"file://{dashboard_path.absolute()}")
        logger.info(f"Opened dashboard in browser: {dashboard_path}")
