# Polymarket Wallet Analyzer

A professional-grade tool for analyzing Polymarket trading wallets, detecting strategies, and generating insights optimized for LLM analysis.

## Features

- **Wallet Scraping**: Fetch complete trading history from Polymarket's API
- **Performance Analysis**: Calculate win rates, P&L, ROI, and risk metrics
- **Strategy Detection**: Identify trading patterns (contrarian, momentum, arbitrage, etc.)
- **LLM-Optimized Export**: Generate structured JSON for AI analysis
- **Interactive Dashboard**: Visualize performance with Plotly charts

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd polymarket-wallet-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
```

## Usage

### Analyze a Wallet

```bash
# Full analysis
python main.py analyze 0xdbade4c82fb72780a0db9a38f821d8671aba9c95

# With custom output
python main.py analyze 0xdbade4c82fb72780a0db9a38f821d8671aba9c95 --output results/
```

### Export Data

```bash
# Export to JSON (LLM-optimized)
python main.py export --format json

# Export to CSV
python main.py export --format csv

# Export to HTML report
python main.py export --format html
```

### Generate Dashboard

```bash
# Generate and open dashboard
python main.py dashboard --open

# Generate without opening
python main.py dashboard
```

## Project Structure

```
polymarket-wallet-analyzer/
├── .env.example          # Environment configuration template
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py         # Configuration management
│   ├── scraper/
│   │   ├── __init__.py
│   │   ├── polymarket_api.py   # API client with rate limiting
│   │   ├── wallet_fetcher.py   # Trade history fetching
│   │   └── data_processor.py   # Data normalization
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── performance.py      # Performance metrics
│   │   ├── strategy_detector.py # Pattern recognition
│   │   └── llm_formatter.py    # LLM export optimization
│   └── visualization/
│       ├── __init__.py
│       ├── dashboard.py        # Plotly dashboard
│       └── exporters.py        # Export utilities
├── tests/
│   └── test_scraper.py
├── outputs/              # Generated reports and data
└── main.py               # CLI entry point
```

## API Reference

### Performance Metrics

- **Win Rate**: Percentage of profitable trades
- **Total P&L**: Realized profit/loss in USD
- **ROI**: Return on investment percentage
- **Sharpe Ratio**: Risk-adjusted return metric
- **Average Hold Time**: Mean duration of positions

### Strategy Detection

The analyzer detects the following patterns:

| Strategy | Description |
|----------|-------------|
| Early Exit | Sells positions before market resolution |
| Contrarian | Buys underdog positions against consensus |
| Momentum | Follows price trends |
| Arbitrage | Takes opposing positions for guaranteed profit |
| Event-Driven | Trades correlated with news events |

## Output Format (LLM-Optimized)

```json
{
  "wallet_summary": {
    "address": "0x...",
    "total_trades": 150,
    "total_volume_usd": 50000,
    "net_pnl_usd": 5000,
    "win_rate": 0.65,
    "active_since": "2024-01-15"
  },
  "top_trades": [...],
  "detected_strategy": "momentum",
  "risk_profile": {...},
  "key_insights": [...]
}
```

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src tests/

# Specific test file
pytest tests/test_scraper.py -v
```

## License

MIT License

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.
