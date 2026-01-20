"""Wallet trading history fetcher with comprehensive data retrieval."""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from ..config import get_logger
from .polymarket_api import PolymarketAPIClient

logger = get_logger(__name__)


@dataclass
class Trade:
    """Normalized trade data structure."""

    id: str
    wallet_address: str
    market_id: str
    market_title: str
    market_category: str
    outcome: str
    side: str  # "buy" or "sell"
    price: Decimal
    size: Decimal
    cost_usd: Decimal
    timestamp: datetime
    transaction_hash: Optional[str] = None
    fee_usd: Decimal = Decimal("0")

    @property
    def is_buy(self) -> bool:
        """Check if this is a buy trade."""
        return self.side.lower() == "buy"

    @property
    def is_sell(self) -> bool:
        """Check if this is a sell trade."""
        return self.side.lower() == "sell"

    def to_dict(self) -> dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            "id": self.id,
            "wallet_address": self.wallet_address,
            "market_id": self.market_id,
            "market_title": self.market_title,
            "market_category": self.market_category,
            "outcome": self.outcome,
            "side": self.side,
            "price": str(self.price),
            "size": str(self.size),
            "cost_usd": str(self.cost_usd),
            "timestamp": self.timestamp.isoformat(),
            "transaction_hash": self.transaction_hash,
            "fee_usd": str(self.fee_usd)
        }


@dataclass
class Position:
    """Normalized position data structure."""

    market_id: str
    market_title: str
    outcome: str
    size: Decimal
    avg_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal

    @property
    def total_pnl(self) -> Decimal:
        """Calculate total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    def to_dict(self) -> dict[str, Any]:
        """Convert position to dictionary."""
        return {
            "market_id": self.market_id,
            "market_title": self.market_title,
            "outcome": self.outcome,
            "size": str(self.size),
            "avg_price": str(self.avg_price),
            "current_price": str(self.current_price),
            "unrealized_pnl": str(self.unrealized_pnl),
            "realized_pnl": str(self.realized_pnl),
            "total_pnl": str(self.total_pnl)
        }


class WalletFetcher:
    """Fetches and normalizes wallet trading history from Polymarket."""

    # Market category mapping
    CATEGORY_KEYWORDS: dict[str, list[str]] = {
        "politics": ["election", "president", "congress", "senate", "trump", "biden", "vote", "governor", "mayor", "democrat", "republican"],
        "crypto": ["bitcoin", "btc", "eth", "ethereum", "crypto", "token", "defi", "solana", "dogecoin"],
        "sports": ["nfl", "nba", "mlb", "soccer", "football", "basketball", "game", "match", "championship", "super bowl", "world cup"],
        "entertainment": ["movie", "oscar", "grammy", "celebrity", "box office", "album", "emmy", "netflix"],
        "finance": ["fed", "interest rate", "inflation", "stock", "market", "gdp", "recession", "s&p", "nasdaq"],
        "science": ["spacex", "nasa", "launch", "ai", "climate", "openai", "gpt", "research"],
        "world": ["ukraine", "russia", "china", "war", "conflict", "treaty", "iran", "israel", "khamenei", "strikes", "military", "supreme leader", "cease"]
    }

    def __init__(self, api_client: Optional[PolymarketAPIClient] = None) -> None:
        """Initialize the wallet fetcher.

        Args:
            api_client: Optional API client instance. Creates new one if not provided.
        """
        self.api = api_client or PolymarketAPIClient()
        self._market_cache: dict[str, dict[str, Any]] = {}

    def _categorize_market(self, title: str) -> str:
        """Categorize a market based on its title.

        Args:
            title: Market title.

        Returns:
            Category string.
        """
        title_lower = title.lower()

        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(keyword in title_lower for keyword in keywords):
                return category

        return "other"

    def _get_market_info(self, market_id: str) -> dict[str, Any]:
        """Get market information with caching.

        Args:
            market_id: Market condition ID.

        Returns:
            Market data dictionary.
        """
        if market_id not in self._market_cache:
            try:
                market_data = self.api.get_market(market_id)
                self._market_cache[market_id] = market_data
            except Exception as e:
                logger.warning(f"Failed to fetch market {market_id}: {e}")
                self._market_cache[market_id] = {
                    "question": "Unknown Market",
                    "slug": market_id
                }

        return self._market_cache[market_id]

    def _normalize_timestamp(self, timestamp: Any) -> datetime:
        """Normalize timestamp to UTC datetime.

        Args:
            timestamp: Timestamp in various formats (int, float, string).

        Returns:
            UTC datetime object.
        """
        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                return timestamp.replace(tzinfo=timezone.utc)
            return timestamp.astimezone(timezone.utc)

        if isinstance(timestamp, str):
            # Try ISO format first
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                return dt.astimezone(timezone.utc)
            except ValueError:
                pass

            # Try Unix timestamp as string
            try:
                ts = float(timestamp)
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            except ValueError:
                pass

        if isinstance(timestamp, (int, float)):
            # Handle milliseconds
            if timestamp > 1e12:
                timestamp = timestamp / 1000
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)

        raise ValueError(f"Unable to parse timestamp: {timestamp}")

    def _parse_trade(self, raw_trade: dict[str, Any], wallet_address: str) -> Trade:
        """Parse raw trade data into Trade object.

        Args:
            raw_trade: Raw trade data from API (data-api format).
            wallet_address: The wallet address being analyzed.

        Returns:
            Normalized Trade object.
        """
        # Data API format fields:
        # proxyWallet, side, asset, conditionId, size, price, timestamp, title, slug, outcome, transactionHash

        market_id = raw_trade.get("conditionId", raw_trade.get("market", "unknown"))

        # Title is directly available in data-api response
        market_title = raw_trade.get("title", "Unknown Market")
        if not market_title or market_title == "Unknown Market":
            # Fallback to fetching market info
            market_info = self._get_market_info(market_id)
            market_title = market_info.get("question", market_info.get("title", "Unknown"))

        # Parse numeric values safely
        price = Decimal(str(raw_trade.get("price", 0)))
        size = Decimal(str(raw_trade.get("size", raw_trade.get("amount", 0))))

        # Calculate cost
        cost_usd = price * size

        # Determine side (data-api uses uppercase BUY/SELL)
        side = raw_trade.get("side", "").lower()
        if not side:
            # Infer from trade type or maker/taker
            if raw_trade.get("type") == "buy" or raw_trade.get("isBuy"):
                side = "buy"
            else:
                side = "sell"

        # Parse timestamp
        timestamp = self._normalize_timestamp(
            raw_trade.get("timestamp", raw_trade.get("createdAt", 0))
        )

        # Get outcome directly from response
        outcome = raw_trade.get("outcome", "Unknown")

        return Trade(
            id=raw_trade.get("transactionHash", raw_trade.get("id", str(hash(str(raw_trade))))),
            wallet_address=wallet_address.lower(),
            market_id=market_id,
            market_title=market_title,
            market_category=self._categorize_market(market_title),
            outcome=outcome,
            side=side,
            price=price,
            size=size,
            cost_usd=cost_usd,
            timestamp=timestamp,
            transaction_hash=raw_trade.get("transactionHash"),
            fee_usd=Decimal(str(raw_trade.get("fee", 0)))
        )

    def _parse_position(self, raw_position: dict[str, Any]) -> Position:
        """Parse raw position data into Position object.

        Args:
            raw_position: Raw position data from API.

        Returns:
            Normalized Position object.
        """
        market_id = raw_position.get("market", raw_position.get("conditionId", "unknown"))
        market_info = self._get_market_info(market_id)
        market_title = market_info.get("question", market_info.get("title", "Unknown"))

        size = Decimal(str(raw_position.get("size", raw_position.get("amount", 0))))
        avg_price = Decimal(str(raw_position.get("avgPrice", raw_position.get("averagePrice", 0))))
        current_price = Decimal(str(raw_position.get("currentPrice", raw_position.get("price", 0))))

        # Calculate P&L
        unrealized_pnl = (current_price - avg_price) * size
        realized_pnl = Decimal(str(raw_position.get("realizedPnl", 0)))

        return Position(
            market_id=market_id,
            market_title=market_title,
            outcome=raw_position.get("outcome", raw_position.get("asset", "Unknown")),
            size=size,
            avg_price=avg_price,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl
        )

    def fetch_all_trades(
        self,
        wallet_address: str,
        batch_size: int = 500
    ) -> list[Trade]:
        """Fetch all trades for a wallet.

        Args:
            wallet_address: Ethereum wallet address.
            batch_size: Number of trades to fetch per request.

        Returns:
            List of normalized Trade objects.
        """
        logger.info(f"Fetching trades for wallet: {wallet_address}")

        all_trades: list[Trade] = []
        offset = 0

        while True:
            logger.debug(f"Fetching trades batch at offset {offset}")

            try:
                raw_trades = self.api.get_wallet_activity(
                    wallet_address,
                    limit=batch_size,
                    offset=offset
                )
            except Exception as e:
                logger.error(f"Failed to fetch trades at offset {offset}: {e}")
                break

            if not raw_trades:
                break

            for raw_trade in raw_trades:
                try:
                    trade = self._parse_trade(raw_trade, wallet_address)
                    all_trades.append(trade)
                except Exception as e:
                    logger.warning(f"Failed to parse trade: {e}")
                    continue

            if len(raw_trades) < batch_size:
                break

            offset += batch_size

        # Sort by timestamp
        all_trades.sort(key=lambda t: t.timestamp)

        logger.info(f"Fetched {len(all_trades)} trades for wallet {wallet_address}")

        return all_trades

    def fetch_positions(self, wallet_address: str) -> list[Position]:
        """Fetch current positions for a wallet.

        Args:
            wallet_address: Ethereum wallet address.

        Returns:
            List of normalized Position objects.
        """
        logger.info(f"Fetching positions for wallet: {wallet_address}")

        try:
            raw_positions = self.api.get_wallet_positions(wallet_address)
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

        positions: list[Position] = []
        for raw_position in raw_positions:
            try:
                position = self._parse_position(raw_position)
                if position.size > 0:
                    positions.append(position)
            except Exception as e:
                logger.warning(f"Failed to parse position: {e}")
                continue

        logger.info(f"Fetched {len(positions)} positions for wallet {wallet_address}")

        return positions

    def fetch_wallet_data(
        self,
        wallet_address: str
    ) -> dict[str, Any]:
        """Fetch complete wallet data including trades and positions.

        Args:
            wallet_address: Ethereum wallet address.

        Returns:
            Dictionary containing trades, positions, and metadata.
        """
        trades = self.fetch_all_trades(wallet_address)
        positions = self.fetch_positions(wallet_address)

        # Calculate basic stats
        first_trade = trades[0].timestamp if trades else None
        last_trade = trades[-1].timestamp if trades else None

        return {
            "wallet_address": wallet_address.lower(),
            "trades": trades,
            "positions": positions,
            "metadata": {
                "total_trades": len(trades),
                "total_positions": len(positions),
                "first_trade": first_trade.isoformat() if first_trade else None,
                "last_trade": last_trade.isoformat() if last_trade else None,
                "fetched_at": datetime.now(timezone.utc).isoformat()
            }
        }
