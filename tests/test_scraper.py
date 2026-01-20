"""Tests for the scraper module."""

import json
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.scraper.data_processor import DataProcessor, ProcessedTrade
from src.scraper.polymarket_api import CacheManager, PolymarketAPIClient, RateLimiter
from src.scraper.wallet_fetcher import Trade, WalletFetcher


class TestRateLimiter:
    """Tests for the RateLimiter class."""

    def test_init(self) -> None:
        """Test rate limiter initialization."""
        limiter = RateLimiter(calls=10, period=60)
        assert limiter.calls == 10
        assert limiter.period == 60
        assert limiter.timestamps == []

    def test_acquire_under_limit(self) -> None:
        """Test acquiring when under rate limit."""
        limiter = RateLimiter(calls=10, period=60)

        # Should not block
        for _ in range(5):
            limiter.acquire()

        assert len(limiter.timestamps) == 5

    def test_acquire_cleans_old_timestamps(self) -> None:
        """Test that old timestamps are cleaned up."""
        limiter = RateLimiter(calls=10, period=1)  # 1 second period

        # Add some timestamps
        limiter.timestamps = [0, 0, 0]  # Very old timestamps

        # Acquire should clean old timestamps
        limiter.acquire()

        # Old timestamps should be removed
        assert len(limiter.timestamps) == 1


class TestTrade:
    """Tests for the Trade dataclass."""

    def test_trade_creation(self) -> None:
        """Test creating a Trade object."""
        trade = Trade(
            id="test-123",
            wallet_address="0x1234567890abcdef",
            market_id="market-1",
            market_title="Test Market",
            market_category="politics",
            outcome="Yes",
            side="buy",
            price=Decimal("0.65"),
            size=Decimal("100"),
            cost_usd=Decimal("65"),
            timestamp=datetime.now(timezone.utc)
        )

        assert trade.id == "test-123"
        assert trade.is_buy is True
        assert trade.is_sell is False

    def test_trade_to_dict(self) -> None:
        """Test converting Trade to dictionary."""
        timestamp = datetime.now(timezone.utc)
        trade = Trade(
            id="test-123",
            wallet_address="0x1234567890abcdef",
            market_id="market-1",
            market_title="Test Market",
            market_category="politics",
            outcome="Yes",
            side="sell",
            price=Decimal("0.75"),
            size=Decimal("50"),
            cost_usd=Decimal("37.5"),
            timestamp=timestamp
        )

        data = trade.to_dict()

        assert data["id"] == "test-123"
        assert data["side"] == "sell"
        assert data["price"] == "0.75"
        assert data["timestamp"] == timestamp.isoformat()


class TestWalletFetcher:
    """Tests for the WalletFetcher class."""

    def test_categorize_market_politics(self) -> None:
        """Test market categorization for politics."""
        fetcher = WalletFetcher()

        assert fetcher._categorize_market("Will Trump win the election?") == "politics"
        assert fetcher._categorize_market("Biden approval rating") == "politics"

    def test_categorize_market_crypto(self) -> None:
        """Test market categorization for crypto."""
        fetcher = WalletFetcher()

        assert fetcher._categorize_market("Bitcoin price above 100k?") == "crypto"
        assert fetcher._categorize_market("ETH merge successful?") == "crypto"

    def test_categorize_market_sports(self) -> None:
        """Test market categorization for sports."""
        fetcher = WalletFetcher()

        assert fetcher._categorize_market("NFL Super Bowl winner") == "sports"
        assert fetcher._categorize_market("NBA Finals MVP") == "sports"

    def test_categorize_market_other(self) -> None:
        """Test market categorization for unknown categories."""
        fetcher = WalletFetcher()

        assert fetcher._categorize_market("Random unknown market") == "other"

    def test_normalize_timestamp_datetime(self) -> None:
        """Test timestamp normalization from datetime."""
        fetcher = WalletFetcher()
        dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        result = fetcher._normalize_timestamp(dt)

        assert result == dt
        assert result.tzinfo == timezone.utc

    def test_normalize_timestamp_iso_string(self) -> None:
        """Test timestamp normalization from ISO string."""
        fetcher = WalletFetcher()

        result = fetcher._normalize_timestamp("2024-01-15T12:00:00Z")

        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_normalize_timestamp_unix(self) -> None:
        """Test timestamp normalization from Unix timestamp."""
        fetcher = WalletFetcher()
        unix_ts = 1705320000  # 2024-01-15 12:00:00 UTC

        result = fetcher._normalize_timestamp(unix_ts)

        assert result.year == 2024
        assert result.tzinfo == timezone.utc

    def test_normalize_timestamp_unix_milliseconds(self) -> None:
        """Test timestamp normalization from Unix milliseconds."""
        fetcher = WalletFetcher()
        unix_ms = 1705320000000  # Milliseconds

        result = fetcher._normalize_timestamp(unix_ms)

        assert result.year == 2024


class TestDataProcessor:
    """Tests for the DataProcessor class."""

    def test_calculate_realized_pnl_buy_only(self) -> None:
        """Test P&L calculation with only buy trades."""
        processor = DataProcessor()

        trades = [
            Trade(
                id="1",
                wallet_address="0x123",
                market_id="m1",
                market_title="Test",
                market_category="other",
                outcome="Yes",
                side="buy",
                price=Decimal("0.5"),
                size=Decimal("100"),
                cost_usd=Decimal("50"),
                timestamp=datetime.now(timezone.utc)
            )
        ]

        processed = processor.calculate_realized_pnl(trades)

        assert len(processed) == 1
        assert processed[0].realized_pnl == Decimal("0")
        assert processed[0].is_closing is False

    def test_calculate_realized_pnl_buy_sell(self) -> None:
        """Test P&L calculation with buy and sell."""
        processor = DataProcessor()

        now = datetime.now(timezone.utc)
        trades = [
            Trade(
                id="1",
                wallet_address="0x123",
                market_id="m1",
                market_title="Test",
                market_category="other",
                outcome="Yes",
                side="buy",
                price=Decimal("0.5"),
                size=Decimal("100"),
                cost_usd=Decimal("50"),
                timestamp=now
            ),
            Trade(
                id="2",
                wallet_address="0x123",
                market_id="m1",
                market_title="Test",
                market_category="other",
                outcome="Yes",
                side="sell",
                price=Decimal("0.7"),
                size=Decimal("100"),
                cost_usd=Decimal("70"),
                timestamp=now
            )
        ]

        processed = processor.calculate_realized_pnl(trades)

        assert len(processed) == 2
        assert processed[0].realized_pnl == Decimal("0")  # Buy has no P&L
        assert processed[1].realized_pnl == Decimal("20")  # Profit: (0.7 - 0.5) * 100
        assert processed[1].is_closing is True

    def test_calculate_realized_pnl_partial_sell(self) -> None:
        """Test P&L calculation with partial sell."""
        processor = DataProcessor()

        now = datetime.now(timezone.utc)
        trades = [
            Trade(
                id="1",
                wallet_address="0x123",
                market_id="m1",
                market_title="Test",
                market_category="other",
                outcome="Yes",
                side="buy",
                price=Decimal("0.5"),
                size=Decimal("100"),
                cost_usd=Decimal("50"),
                timestamp=now
            ),
            Trade(
                id="2",
                wallet_address="0x123",
                market_id="m1",
                market_title="Test",
                market_category="other",
                outcome="Yes",
                side="sell",
                price=Decimal("0.6"),
                size=Decimal("50"),  # Only sell half
                cost_usd=Decimal("30"),
                timestamp=now
            )
        ]

        processed = processor.calculate_realized_pnl(trades)

        assert len(processed) == 2
        # P&L on sold portion: (0.6 - 0.5) * 50 = 5
        assert processed[1].realized_pnl == Decimal("5")
        # Should still have 50 shares
        assert processed[1].position_size_after == Decimal("50")

    def test_to_dataframe(self) -> None:
        """Test converting trades to DataFrame."""
        processor = DataProcessor()

        trades = [
            Trade(
                id="1",
                wallet_address="0x123",
                market_id="m1",
                market_title="Test",
                market_category="politics",
                outcome="Yes",
                side="buy",
                price=Decimal("0.5"),
                size=Decimal("100"),
                cost_usd=Decimal("50"),
                timestamp=datetime.now(timezone.utc)
            )
        ]

        df = processor.to_dataframe(trades)

        assert len(df) == 1
        assert df.iloc[0]["market_category"] == "politics"
        assert df.iloc[0]["price"] == 0.5


class TestPolymarketAPIClient:
    """Tests for the PolymarketAPIClient class."""

    @patch("src.scraper.polymarket_api.requests.Session")
    def test_client_initialization(self, mock_session: MagicMock) -> None:
        """Test API client initialization."""
        client = PolymarketAPIClient()

        assert client.base_url is not None
        assert client.rate_limiter is not None

    @patch("src.scraper.polymarket_api.requests.Session")
    def test_get_request(self, mock_session_class: MagicMock) -> None:
        """Test GET request."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": "test"}
        mock_response.raise_for_status = MagicMock()
        mock_session.request.return_value = mock_response
        mock_session_class.return_value = mock_session

        client = PolymarketAPIClient()
        client.session = mock_session

        result = client.get("test-endpoint", use_cache=False)

        assert result == {"data": "test"}


class TestCacheManager:
    """Tests for the CacheManager class."""

    def test_cache_key_generation(self, tmp_path) -> None:
        """Test cache key generation."""
        cache = CacheManager(tmp_path, ttl=3600)

        key1 = cache._get_cache_key("http://example.com/api", {"a": 1})
        key2 = cache._get_cache_key("http://example.com/api", {"a": 1})
        key3 = cache._get_cache_key("http://example.com/api", {"a": 2})

        assert key1 == key2  # Same URL and params
        assert key1 != key3  # Different params

    def test_cache_set_get(self, tmp_path) -> None:
        """Test cache set and get."""
        cache = CacheManager(tmp_path, ttl=3600)
        url = "http://example.com/api"
        params = {"test": True}
        data = {"result": "cached"}

        # Set cache
        cache.set(url, params, data)

        # Get cache
        result = cache.get(url, params)

        assert result == data

    def test_cache_miss(self, tmp_path) -> None:
        """Test cache miss."""
        cache = CacheManager(tmp_path, ttl=3600)

        result = cache.get("http://nonexistent.com", None)

        assert result is None


# Integration test placeholder
class TestIntegration:
    """Integration tests (require network access)."""

    @pytest.mark.skip(reason="Requires network access")
    def test_fetch_real_wallet(self) -> None:
        """Test fetching real wallet data."""
        fetcher = WalletFetcher()
        wallet = "0xdbade4c82fb72780a0db9a38f821d8671aba9c95"

        data = fetcher.fetch_wallet_data(wallet)

        assert "trades" in data
        assert "positions" in data
