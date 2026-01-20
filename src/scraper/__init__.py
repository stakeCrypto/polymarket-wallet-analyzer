"""Scraper module for Polymarket data fetching."""

from .data_processor import DataProcessor
from .polymarket_api import PolymarketAPIClient
from .wallet_fetcher import WalletFetcher

__all__ = ["PolymarketAPIClient", "WalletFetcher", "DataProcessor"]
