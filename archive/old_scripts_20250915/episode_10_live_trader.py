#!/usr/bin/env python3
"""
Episode 10 Live Trading System
Waits for new market data and runs inference only when needed
"""

import sys
import os
import time
import signal
import logging
from datetime import datetime, timezone
from pathlib import Path
import threading
import queue

sys.path.append(str(Path(__file__).parent))

from swt_validation.fixed_checkpoint_loader import load_checkpoint_with_proper_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Episode10LiveTrader:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.running = True
        self.model = None
        self.last_price = None
        self.inference_count = 0

        # Market hours (Forex is 24/5, but we'll simulate some quiet periods)
        self.market_open_hour = 0  # 24/5 for Forex
        self.market_close_hour = 24

        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    def load_model(self):
        """Load Episode 10 checkpoint"""
        logger.info(f"🚀 Loading Episode 10 model from {self.checkpoint_path}")
        try:
            checkpoint = load_checkpoint_with_proper_config(self.checkpoint_path)
            self.model = checkpoint['networks']
            self.model.eval()

            config = checkpoint['config']
            logger.info(f"✅ Model loaded successfully!")
            logger.info(f"   Architecture: hidden_dim={config.get('hidden_dim')}, support_size={config.get('support_size')}")
            logger.info(f"   Networks: Repr, Dynamics, Policy, Value, Chance")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False

    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now(timezone.utc)
        hour = now.hour
        weekday = now.weekday()  # Monday = 0, Sunday = 6

        # Forex is closed on weekends (Saturday-Sunday)
        if weekday >= 5:  # Saturday = 5, Sunday = 6
            return False

        # For demo purposes, let's say market is "active" during certain hours
        # In reality, Forex is 24/5, but we'll simulate periods
        return self.market_open_hour <= hour < self.market_close_hour

    def simulate_price_feed(self):
        """Simulate receiving new price data"""
        # In real implementation, this would connect to:
        # - MetaTrader 5 API
        # - Interactive Brokers API
        # - Forex data feed
        # - WebSocket price stream

        import random
        base_price = 150.0  # GBPJPY around 150

        while self.running:
            if self.is_market_open():
                # Simulate price movement
                change = random.gauss(0, 0.001)  # Small random walk
                new_price = base_price + change
                base_price = new_price

                yield {
                    'timestamp': datetime.now(timezone.utc),
                    'symbol': 'GBPJPY',
                    'price': new_price,
                    'bid': new_price - 0.001,
                    'ask': new_price + 0.001
                }

                # New price every 1 second during market hours
                time.sleep(1)
            else:
                # Market closed - check every 5 minutes
                logger.info("🌙 Market closed - sleeping until market opens...")
                time.sleep(300)  # 5 minutes

    def run_inference(self, price_data: dict):
        """Run Episode 10 inference on new price data"""
        try:
            import torch
            import numpy as np

            # Create mock features for inference
            # In real implementation, this would:
            # 1. Update price history buffer
            # 2. Compute WST features for recent window
            # 3. Extract position features

            # Mock 137 features (128 WST + 9 position)
            features = torch.randn(1, 137)

            with torch.no_grad():
                # Run Episode 10 inference
                hidden = self.model.representation_network(features)
                latent = self.model.chance_encoder(features)
                policy = self.model.policy_network(hidden, latent)
                value = self.model.value_network(hidden, latent)

                # Extract action probabilities
                action_probs = torch.softmax(policy, dim=-1).numpy()[0]
                actions = ['HOLD', 'BUY', 'SELL', 'CLOSE']

                # Log inference result
                predicted_action = actions[np.argmax(action_probs)]
                confidence = float(np.max(action_probs))

                logger.info(f"💡 INFERENCE #{self.inference_count + 1}")
                logger.info(f"   Price: {price_data['price']:.5f}")
                logger.info(f"   Action: {predicted_action} (confidence: {confidence:.2%})")
                logger.info(f"   Hidden state: {hidden.shape}")

                self.inference_count += 1

                # In real implementation, this would:
                # - Send order to broker if action != HOLD
                # - Update position management
                # - Log trade decisions

        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")

    def run(self):
        """Main trading loop"""
        logger.info("🎯 Episode 10 Live Trading System Starting...")

        # Load model
        if not self.load_model():
            logger.error("Failed to load model, exiting...")
            return

        logger.info("🔄 Starting live trading loop...")
        logger.info("   - Monitoring market hours")
        logger.info("   - Waiting for new price data")
        logger.info("   - Running inference only when needed")
        logger.info("   - Press Ctrl+C to stop")

        try:
            # Start price feed simulation
            price_feed = self.simulate_price_feed()

            for price_data in price_feed:
                if not self.running:
                    break

                # Run inference on new price
                self.run_inference(price_data)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Trading loop error: {e}")
        finally:
            logger.info(f"🏁 Shutting down. Total inferences: {self.inference_count}")

def main():
    checkpoint_path = "checkpoints/episode_10_best.pth"

    trader = Episode10LiveTrader(checkpoint_path)
    trader.run()

if __name__ == "__main__":
    main()