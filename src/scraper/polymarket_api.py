"""Polymarket API client with rate limiting and retry logic."""

import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import get_config, get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, calls: int, period: int) -> None:
        """Initialize rate limiter.

        Args:
            calls: Maximum number of calls allowed in the period.
            period: Time period in seconds.
        """
        self.calls = calls
        self.period = period
        self.timestamps: list[float] = []

    def acquire(self) -> None:
        """Acquire permission to make an API call, blocking if necessary."""
        now = time.time()

        # Remove timestamps outside the window
        self.timestamps = [
            ts for ts in self.timestamps
            if now - ts < self.period
        ]

        if len(self.timestamps) >= self.calls:
            # Need to wait
            sleep_time = self.timestamps[0] + self.period - now
            if sleep_time > 0:
                logger.debug(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
                self.timestamps = self.timestamps[1:]

        self.timestamps.append(time.time())


class CacheManager:
    """Simple file-based cache for API responses."""

    def __init__(self, cache_dir: Path, ttl: int) -> None:
        """Initialize cache manager.

        Args:
            cache_dir: Directory to store cache files.
            ttl: Time-to-live in seconds for cache entries.
        """
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, url: str, params: Optional[dict[str, Any]]) -> str:
        """Generate a unique cache key for the request."""
        key_data = f"{url}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{key}.json"

    def get(self, url: str, params: Optional[dict[str, Any]] = None) -> Optional[Any]:
        """Retrieve cached response if valid.

        Args:
            url: Request URL.
            params: Request parameters.

        Returns:
            Cached data if valid, None otherwise.
        """
        key = self._get_cache_key(url, params)
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path) as f:
                cached = json.load(f)

            cached_time = datetime.fromisoformat(cached["timestamp"])
            if datetime.utcnow() - cached_time > timedelta(seconds=self.ttl):
                cache_path.unlink()
                return None

            logger.debug(f"Cache hit for {url}")
            return cached["data"]
        except (json.JSONDecodeError, KeyError, ValueError):
            cache_path.unlink(missing_ok=True)
            return None

    def set(self, url: str, params: Optional[dict[str, Any]], data: Any) -> None:
        """Store response in cache.

        Args:
            url: Request URL.
            params: Request parameters.
            data: Response data to cache.
        """
        key = self._get_cache_key(url, params)
        cache_path = self._get_cache_path(key)

        with open(cache_path, "w") as f:
            json.dump({
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            }, f)

        logger.debug(f"Cached response for {url}")


class PolymarketAPIClient:
    """API client for Polymarket with rate limiting, caching, and retry logic."""

    def __init__(self) -> None:
        """Initialize the API client."""
        self.config = get_config()
        self.base_url = self.config.api.base_url
        self.clob_url = self.config.api.clob_url

        # Setup rate limiter
        self.rate_limiter = RateLimiter(
            self.config.api.rate_limit_calls,
            self.config.api.rate_limit_period
        )

        # Setup cache
        self.cache: Optional[CacheManager] = None
        if self.config.cache.enabled:
            self.cache = CacheManager(
                self.config.cache.directory,
                self.config.cache.ttl
            )

        # Setup session with retry logic
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry configuration."""
        session = requests.Session()

        retry_strategy = Retry(
            total=self.config.api.max_retries,
            backoff_factor=self.config.api.retry_backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        session.headers.update({
            "User-Agent": "PolymarketWalletAnalyzer/1.0",
            "Accept": "application/json"
        })

        return session

    def _request(
        self,
        method: str,
        url: str,
        params: Optional[dict[str, Any]] = None,
        json_data: Optional[dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Any:
        """Make an API request with rate limiting and caching.

        Args:
            method: HTTP method (GET, POST).
            url: Full URL or endpoint.
            params: Query parameters.
            json_data: JSON body for POST requests.
            use_cache: Whether to use cache for this request.

        Returns:
            Parsed JSON response.

        Raises:
            requests.RequestException: If the request fails.
        """
        # Check cache for GET requests
        if method == "GET" and use_cache and self.cache:
            cached = self.cache.get(url, params)
            if cached is not None:
                return cached

        # Apply rate limiting
        self.rate_limiter.acquire()

        logger.debug(f"Making {method} request to {url}")

        response = self.session.request(
            method=method,
            url=url,
            params=params,
            json=json_data,
            timeout=self.config.api.timeout
        )

        response.raise_for_status()
        data = response.json()

        # Cache GET responses
        if method == "GET" and use_cache and self.cache:
            self.cache.set(url, params, data)

        return data

    def get(
        self,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Any:
        """Make a GET request to the API.

        Args:
            endpoint: API endpoint (will be appended to base_url).
            params: Query parameters.
            use_cache: Whether to use cache for this request.

        Returns:
            Parsed JSON response.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        return self._request("GET", url, params=params, use_cache=use_cache)

    def post(
        self,
        endpoint: str,
        json_data: dict[str, Any],
        params: Optional[dict[str, Any]] = None
    ) -> Any:
        """Make a POST request to the API.

        Args:
            endpoint: API endpoint (will be appended to base_url).
            json_data: JSON body data.
            params: Query parameters.

        Returns:
            Parsed JSON response.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        return self._request("POST", url, params=params, json_data=json_data)

    def get_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        active: bool = True
    ) -> list[dict[str, Any]]:
        """Fetch list of markets.

        Args:
            limit: Maximum number of markets to fetch.
            offset: Offset for pagination.
            active: Whether to fetch only active markets.

        Returns:
            List of market data dictionaries.
        """
        params = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower()
        }
        return self.get("markets", params=params)

    def get_market(self, market_id: str) -> dict[str, Any]:
        """Fetch a specific market by ID.

        Args:
            market_id: The market's condition ID.

        Returns:
            Market data dictionary.
        """
        return self.get(f"markets/{market_id}")

    def get_trades(
        self,
        market_id: Optional[str] = None,
        maker: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[dict[str, Any]]:
        """Fetch trades with optional filters.

        Args:
            market_id: Filter by market ID.
            maker: Filter by maker address.
            limit: Maximum number of trades to fetch.
            offset: Offset for pagination.

        Returns:
            List of trade data dictionaries.
        """
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset
        }

        if market_id:
            params["market"] = market_id
        if maker:
            params["maker"] = maker

        return self.get("trades", params=params)

    def get_wallet_activity(
        self,
        wallet_address: str,
        limit: int = 500,
        offset: int = 0
    ) -> list[dict[str, Any]]:
        """Fetch wallet trading activity.

        Args:
            wallet_address: Ethereum wallet address.
            limit: Maximum number of activities to fetch.
            offset: Offset for pagination.

        Returns:
            List of activity data dictionaries.
        """
        params = {
            "user": wallet_address.lower(),
            "limit": limit,
            "offset": offset
        }
        return self.get("activity", params=params)

    def get_wallet_positions(self, wallet_address: str) -> list[dict[str, Any]]:
        """Fetch current positions for a wallet.

        Args:
            wallet_address: Ethereum wallet address.

        Returns:
            List of position data dictionaries.
        """
        params = {"user": wallet_address.lower()}
        return self.get("positions", params=params)

    def get_price_history(
        self,
        token_id: str,
        interval: str = "1d",
        fidelity: int = 60
    ) -> list[dict[str, Any]]:
        """Fetch price history for a token.

        Args:
            token_id: The token ID.
            interval: Time interval (1h, 1d, etc.).
            fidelity: Data fidelity in minutes.

        Returns:
            List of price history data points.
        """
        params = {
            "tokenId": token_id,
            "interval": interval,
            "fidelity": fidelity
        }
        return self.get("prices-history", params=params)
