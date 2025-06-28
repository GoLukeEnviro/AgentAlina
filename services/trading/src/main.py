import ccxt
import websockets
import aiohttp
import asyncio
import logging
import os
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MCP client for monitor-mcp
MCP_MONITOR_ENDPOINT = "http://monitor-mcp:8002"

# Exchange configuration (example with Binance)
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "binance")
API_KEY = os.getenv("API_KEY", "your_api_key")
API_SECRET = os.getenv("API_SECRET", "your_api_secret")

class TradingBot:
    def __init__(self):
        self.exchange = getattr(ccxt, EXCHANGE_ID)({
            'apiKey': API_KEY,
            'secret': API_SECRET,
        })
        self.websocket_url = "wss://stream.binance.com:9443/ws"
        self.symbol = "BTC/USDT"
        self.alert_thresholds = {
            "profit": 0.05,  # 5% profit
            "loss": -0.03    # 3% loss
        }
        logger.info(f"Initialized trading bot for {EXCHANGE_ID} with symbol {self.symbol}")

    async def stream_pnl(self):
        """Stream P&L data via WebSocket from the exchange."""
        uri = f"{self.websocket_url}/{self.symbol.lower().replace('/', '')}@trade"
        async with websockets.connect(uri) as websocket:
            logger.info(f"Connected to WebSocket stream for {self.symbol}")
            while True:
                try:
                    data = await websocket.recv()
                    trade_data = json.loads(data)
                    price = trade_data.get("p")
                    if price:
                        await self.check_alerts(float(price))
                except Exception as e:
                    logger.error(f"Error in WebSocket stream: {e}")
                    await asyncio.sleep(5)
                    break

    async def check_alerts(self, current_price):
        """Check if current price triggers any P&L alerts based on thresholds."""
        # Placeholder for position data (in a real scenario, fetch from exchange or local store)
        entry_price = 50000  # Example entry price for BTC/USDT
        percentage_change = (current_price - entry_price) / entry_price
        
        if percentage_change >= self.alert_thresholds["profit"]:
            alert_msg = f"Profit alert: {self.symbol} price increased by {percentage_change*100:.2f}% to {current_price}"
            logger.info(alert_msg)
            await self.send_alert("profit_alert", alert_msg)
        elif percentage_change <= self.alert_thresholds["loss"]:
            alert_msg = f"Loss alert: {self.symbol} price decreased by {percentage_change*100:.2f}% to {current_price}"
            logger.warning(alert_msg)
            await self.send_alert("loss_alert", alert_msg)

    async def send_alert(self, alert_type, message):
        """Send alert to monitor-mcp for processing."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "type": alert_type,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
            try:
                async with session.post(f"{MCP_MONITOR_ENDPOINT}/define_alert", json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Alert sent to monitor-mcp: {alert_type}")
                    else:
                        logger.error(f"Failed to send alert to monitor-mcp: {response.status}")
            except Exception as e:
                logger.error(f"Error sending alert to monitor-mcp: {e}")

    async def execute_trade(self, side, amount, price=None):
        """Execute a trade on the exchange (buy or sell)."""
        try:
            order_type = 'market' if price is None else 'limit'
            if order_type == 'market':
                order = self.exchange.create_market_order(self.symbol, side, amount)
            else:
                order = self.exchange.create_limit_order(self.symbol, side, amount, price)
            logger.info(f"Executed {side} order for {amount} {self.symbol} at {order_type} price")
            return order
        except Exception as e:
            logger.error(f"Error executing {side} order for {self.symbol}: {e}")
            return None

def main():
    """Main function to run the trading service."""
    logger.info("Starting Trading Service...")
    bot = TradingBot()
    
    # Start WebSocket stream for P&L monitoring
    loop = asyncio.get_event_loop()
    loop.create_task(bot.stream_pnl())
    
    logger.info("Trading Service initialized. Listening for market data...")
    
    # Keep the service running
    loop.run_forever()

if __name__ == "__main__":
    main()
