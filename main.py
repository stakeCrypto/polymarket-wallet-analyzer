#!/usr/bin/env python3
"""
Polymarket Wallet Analyzer CLI

A professional tool for analyzing Polymarket trading wallets.

Usage:
    python main.py                         # Interactive mode
    python main.py analyze <wallet_address>
    python main.py export --format json
    python main.py dashboard --open
"""

import sys
from pathlib import Path
from typing import Optional

import click

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.llm_formatter import LLMFormatter
from src.analysis.performance import PerformanceAnalyzer
from src.analysis.strategy_detector import StrategyDetector
from src.config import get_config, get_logger
from src.scraper.data_processor import DataProcessor
from src.scraper.polymarket_api import PolymarketAPIClient
from src.scraper.wallet_fetcher import WalletFetcher
from src.visualization.dashboard import Dashboard
from src.visualization.exporters import CSVExporter, HTMLExporter, JSONExporter

logger = get_logger(__name__)

# Store analysis results for export commands
_analysis_cache: dict = {}


@click.group()
@click.version_option(version="1.0.0", prog_name="Polymarket Wallet Analyzer")
def cli() -> None:
    """Polymarket Wallet Analyzer - Professional trading analysis toolkit."""
    pass


@cli.command()
@click.argument("wallet_address")
@click.option("--output", "-o", type=click.Path(), help="Output directory for exports")
@click.option("--format", "-f", "export_format", type=click.Choice(["json", "csv", "html", "all"]),
              default="json", help="Export format")
@click.option("--no-dashboard", is_flag=True, help="Skip dashboard generation")
@click.option("--open-dashboard", is_flag=True, help="Open dashboard in browser")
def analyze(
    wallet_address: str,
    output: Optional[str],
    export_format: str,
    no_dashboard: bool,
    open_dashboard: bool
) -> None:
    """Analyze a Polymarket wallet.

    WALLET_ADDRESS: Ethereum wallet address to analyze (0x...)
    """
    click.echo(f"\n{'='*60}")
    click.echo(f"  Polymarket Wallet Analyzer v1.0.0")
    click.echo(f"{'='*60}\n")

    # Validate wallet address
    if not wallet_address.startswith("0x") or len(wallet_address) != 42:
        click.echo(click.style("Error: Invalid wallet address format", fg="red"))
        click.echo("Address should be a 42-character hex string starting with 0x")
        sys.exit(1)

    output_dir = Path(output) if output else get_config().export.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Fetch wallet data
        click.echo(click.style("Step 1/5: Fetching wallet data...", fg="cyan"))
        api_client = PolymarketAPIClient()
        wallet_fetcher = WalletFetcher(api_client)

        wallet_data = wallet_fetcher.fetch_wallet_data(wallet_address)
        trades = wallet_data["trades"]
        positions = wallet_data["positions"]
        redemptions = wallet_data.get("redemptions", [])

        click.echo(f"  Found {len(trades)} trades, {len(redemptions)} redemptions, {len(positions)} open positions")

        if not trades:
            click.echo(click.style("\nNo trades found for this wallet.", fg="yellow"))
            click.echo("The wallet may be empty or the API may be unavailable.")
            sys.exit(0)

        # Step 2: Process trades
        click.echo(click.style("\nStep 2/5: Processing trades...", fg="cyan"))
        data_processor = DataProcessor()
        processed_trades = data_processor.calculate_realized_pnl(trades)
        df = data_processor.processed_to_dataframe(processed_trades)

        click.echo(f"  Processed {len(processed_trades)} trades")

        # Step 3: Analyze performance
        click.echo(click.style("\nStep 3/5: Analyzing performance...", fg="cyan"))
        performance_analyzer = PerformanceAnalyzer()
        metrics = performance_analyzer.analyze(processed_trades)

        click.echo(f"  Win Rate: {metrics.win_rate:.1%}")
        click.echo(f"  Total P&L: ${metrics.total_realized_pnl:,.2f}")
        click.echo(f"  ROI: {metrics.roi_percent:.1f}%")

        # Step 4: Detect strategy
        click.echo(click.style("\nStep 4/5: Detecting trading strategy...", fg="cyan"))
        strategy_detector = StrategyDetector()
        strategy = strategy_detector.detect(processed_trades)

        click.echo(f"  Primary Strategy: {strategy.primary_strategy.value.replace('_', ' ').title()}")
        click.echo(f"  Risk Level: {strategy.risk_profile.get('risk_level', 'unknown').title()}")

        # Step 5: Generate exports
        click.echo(click.style("\nStep 5/5: Generating exports...", fg="cyan"))

        # Generate LLM export (includes redemption P&L)
        llm_formatter = LLMFormatter()
        llm_export = llm_formatter.format(wallet_address, trades, positions, processed_trades, redemptions)

        # Cache for export command
        _analysis_cache["llm_export"] = llm_export
        _analysis_cache["wallet_address"] = wallet_address
        _analysis_cache["trades"] = trades
        _analysis_cache["processed_trades"] = processed_trades
        _analysis_cache["metrics"] = metrics
        _analysis_cache["strategy"] = strategy
        _analysis_cache["df"] = df

        # Export based on format
        exported_files = []

        if export_format in ["json", "all"]:
            json_exporter = JSONExporter(output_dir)
            path = json_exporter.export_llm_format(llm_export, wallet_address)
            exported_files.append(("JSON (LLM format)", path))

        if export_format in ["csv", "all"]:
            csv_exporter = CSVExporter(output_dir)
            path = csv_exporter.export_processed_trades(processed_trades, wallet_address)
            exported_files.append(("CSV (trades)", path))
            path = csv_exporter.export_summary(wallet_address, metrics, strategy)
            exported_files.append(("CSV (summary)", path))

        if export_format in ["html", "all"]:
            html_exporter = HTMLExporter(output_dir)
            path = html_exporter.export(llm_export, wallet_address)
            exported_files.append(("HTML report", path))

        # Generate dashboard
        if not no_dashboard:
            dashboard = Dashboard(output_dir)
            dashboard_path = dashboard.generate(wallet_address, df, metrics, strategy)
            exported_files.append(("Dashboard", dashboard_path))

            if open_dashboard:
                dashboard.open_dashboard(dashboard_path)

        # Print summary
        click.echo(click.style("\n" + "="*60, fg="green"))
        click.echo(click.style("  Analysis Complete!", fg="green", bold=True))
        click.echo(click.style("="*60 + "\n", fg="green"))

        click.echo(click.style("Summary:", fg="white", bold=True))
        click.echo(f"  Wallet: {wallet_address[:10]}...{wallet_address[-4:]}")
        click.echo(f"  Total Trades: {metrics.total_trades}")
        click.echo(f"  Redemptions: {len(redemptions)}")
        click.echo(f"  Total Volume: ${metrics.total_volume_usd:,.0f}")

        # Show P&L breakdown (trades + redemptions + unrealized)
        total_pnl = llm_export.wallet_summary.get("total_pnl_usd", metrics.total_realized_pnl)
        redemption_pnl = llm_export.wallet_summary.get("redemption_pnl_usd", 0)
        unrealized_pnl = llm_export.wallet_summary.get("unrealized_pnl_usd", 0)

        pnl_color = "green" if total_pnl >= 0 else "red"
        click.echo(f"  Trade P&L: ${metrics.total_realized_pnl:+,.2f}")
        if redemption_pnl != 0:
            redemption_color = "green" if redemption_pnl >= 0 else "red"
            click.echo(f"  Redemption P&L: " + click.style(f"${redemption_pnl:+,.2f}", fg=redemption_color))
        if unrealized_pnl != 0:
            unrealized_color = "green" if unrealized_pnl >= 0 else "red"
            click.echo(f"  Unrealized P&L: " + click.style(f"${unrealized_pnl:+,.2f}", fg=unrealized_color) + f" ({len(positions)} positions)")
        click.echo(f"  Total P&L: " + click.style(f"${total_pnl:+,.2f}", fg=pnl_color))
        click.echo(f"  Win Rate: {metrics.win_rate:.1%}")
        click.echo(f"  Strategy: {strategy.primary_strategy.value.replace('_', ' ').title()}")

        click.echo(click.style("\nExported Files:", fg="white", bold=True))
        for name, path in exported_files:
            click.echo(f"  {name}: {path}")

        # Print key insights
        if llm_export.key_insights:
            click.echo(click.style("\nKey Insights:", fg="white", bold=True))
            for i, insight in enumerate(llm_export.key_insights[:5], 1):
                click.echo(f"  {i}. {insight}")

        click.echo()

    except Exception as e:
        logger.exception("Analysis failed")
        click.echo(click.style(f"\nError: {str(e)}", fg="red"))
        click.echo("Check the log file for details.")
        sys.exit(1)


@cli.command()
@click.option("--format", "-f", "export_format", type=click.Choice(["json", "csv", "html"]),
              default="json", help="Export format")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def export(export_format: str, output: Optional[str]) -> None:
    """Export analysis results to specified format.

    Requires running 'analyze' command first.
    """
    if not _analysis_cache:
        click.echo(click.style("Error: No analysis data available.", fg="red"))
        click.echo("Run 'analyze' command first to generate data.")
        sys.exit(1)

    output_dir = get_config().export.output_dir

    try:
        wallet_address = _analysis_cache["wallet_address"]

        if export_format == "json":
            exporter = JSONExporter(output_dir)
            path = exporter.export_llm_format(
                _analysis_cache["llm_export"],
                wallet_address,
                output
            )
        elif export_format == "csv":
            exporter = CSVExporter(output_dir)
            path = exporter.export_processed_trades(
                _analysis_cache["processed_trades"],
                wallet_address,
                output
            )
        else:  # html
            exporter = HTMLExporter(output_dir)
            path = exporter.export(
                _analysis_cache["llm_export"],
                wallet_address,
                output
            )

        click.echo(click.style(f"Exported to: {path}", fg="green"))

    except Exception as e:
        logger.exception("Export failed")
        click.echo(click.style(f"Error: {str(e)}", fg="red"))
        sys.exit(1)


@cli.command()
@click.option("--open", "open_browser", is_flag=True, help="Open dashboard in browser")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def dashboard(open_browser: bool, output: Optional[str]) -> None:
    """Generate interactive dashboard.

    Requires running 'analyze' command first.
    """
    if not _analysis_cache:
        click.echo(click.style("Error: No analysis data available.", fg="red"))
        click.echo("Run 'analyze' command first to generate data.")
        sys.exit(1)

    output_dir = Path(output).parent if output else get_config().export.output_dir

    try:
        dash = Dashboard(output_dir)
        dashboard_path = dash.generate(
            _analysis_cache["wallet_address"],
            _analysis_cache["df"],
            _analysis_cache["metrics"],
            _analysis_cache["strategy"]
        )

        click.echo(click.style(f"Dashboard generated: {dashboard_path}", fg="green"))

        if open_browser:
            dash.open_dashboard(dashboard_path)
            click.echo("Opened in browser.")

    except Exception as e:
        logger.exception("Dashboard generation failed")
        click.echo(click.style(f"Error: {str(e)}", fg="red"))
        sys.exit(1)


@cli.command()
@click.argument("wallet_address")
def quick(wallet_address: str) -> None:
    """Quick analysis with compact output.

    WALLET_ADDRESS: Ethereum wallet address to analyze (0x...)
    """
    click.echo("Analyzing wallet...")

    try:
        api_client = PolymarketAPIClient()
        wallet_fetcher = WalletFetcher(api_client)
        wallet_data = wallet_fetcher.fetch_wallet_data(wallet_address)

        if not wallet_data["trades"]:
            click.echo("No trades found.")
            return

        data_processor = DataProcessor()
        processed_trades = data_processor.calculate_realized_pnl(wallet_data["trades"])
        redemptions = wallet_data.get("redemptions", [])

        llm_formatter = LLMFormatter()
        compact = llm_formatter.format_compact(
            wallet_address,
            wallet_data["trades"],
            wallet_data["positions"],
            processed_trades,
            redemptions
        )

        click.echo(f"\n{'='*40}")
        click.echo(f"Wallet: {compact['wallet']}")
        click.echo(f"{'='*40}")
        click.echo(f"Trades: {compact['summary']['trades']}")
        click.echo(f"Redemptions: {compact.get('redemptions', 0)}")
        click.echo(f"Open Positions: {compact.get('positions', 0)}")
        click.echo(f"Volume: {compact['summary']['volume']}")
        click.echo(f"Total P&L: {compact['summary']['pnl']}")
        click.echo(f"  Trade P&L: {compact['summary'].get('trade_pnl', compact['summary']['pnl'])}")
        click.echo(f"  Redemption P&L: {compact['summary'].get('redemption_pnl', '$0')}")
        click.echo(f"  Unrealized P&L: {compact['summary'].get('unrealized_pnl', '$0')}")
        click.echo(f"Win Rate: {compact['summary']['win_rate']}")
        click.echo(f"ROI: {compact['summary']['roi']}")
        click.echo(f"Strategy: {compact['strategy']}")
        click.echo(f"Risk: {compact['risk_level']}")
        click.echo(f"\nInsight: {compact['top_insight']}")

    except Exception as e:
        click.echo(click.style(f"Error: {str(e)}", fg="red"))
        sys.exit(1)


@cli.command()
def interactive() -> None:
    """Launch interactive mode with menu-driven interface."""
    from src.interactive import InteractiveAnalyzer
    analyzer = InteractiveAnalyzer()
    analyzer.run()


if __name__ == "__main__":
    # If no arguments provided, launch interactive mode
    if len(sys.argv) == 1:
        from src.interactive import InteractiveAnalyzer
        analyzer = InteractiveAnalyzer()
        analyzer.run()
    else:
        cli()
