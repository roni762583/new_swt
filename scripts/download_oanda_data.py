#!/usr/bin/env python3
"""
SWT Oanda Data Downloader
Memory-efficient historical data downloader for the new SWT system
Adapted from the original comprehensive downloader for SWT-specific needs
"""

import os
import sys
import v20
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dateutil import parser
import time
import logging
from typing import Optional, Union, List
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SWTOandaDownloader:
    """
    Memory-efficient Oanda data downloader specifically for SWT system
    Downloads data in <5000 bar batches and saves incrementally to CSV
    """
    
    MAX_CANDLES_PER_REQUEST = 4999  # Stay just under Oanda's 5000 limit
    RATE_LIMIT_DELAY = 0.6  # Conservative rate limiting
    
    def __init__(self, api_key: str, account_id: str, environment: str = 'live'):
        """
        Initialize downloader with Oanda credentials
        
        Args:
            api_key: Oanda API key
            account_id: Oanda account ID  
            environment: 'live' or 'practice'
        """
        self.api_key = api_key
        self.account_id = account_id
        self.environment = environment
        self.api_context = None
        self._create_api_context()
    
    def _create_api_context(self):
        """Create Oanda API context"""
        try:
            if self.environment == 'live':
                hostname = "api-fxtrade.oanda.com"
            else:
                hostname = "api-fxpractice.oanda.com"
            
            self.api_context = v20.Context(
                hostname=hostname,
                port="443",
                token=self.api_key
            )
            logger.info(f"✅ Oanda API context created ({self.environment})")
        except Exception as e:
            logger.error(f"❌ Failed to create API context: {e}")
            raise
    
    def download_historical_csv(
        self,
        instrument: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        output_path: str,
        granularity: str = 'M1'
    ) -> bool:
        """
        Download historical data and save incrementally to CSV
        
        Args:
            instrument: Forex pair (e.g., 'GBP_JPY')
            start_date: Start date ('2022-01-01' or datetime)
            end_date: End date ('2025-08-07' or datetime) 
            output_path: Path for output CSV file
            granularity: Timeframe (default: 'M1')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Parse dates
            if isinstance(start_date, str):
                start_dt = parser.parse(start_date).replace(tzinfo=timezone.utc)
            else:
                start_dt = start_date
            
            if isinstance(end_date, str):
                end_dt = parser.parse(end_date).replace(tzinfo=timezone.utc)
            else:
                end_dt = end_date
            
            logger.info(f"🚀 Starting {instrument} {granularity} download")
            logger.info(f"📅 Date range: {start_dt.date()} to {end_dt.date()}")
            logger.info(f"💾 Output file: {output_path}")
            
            # Estimate total candles for progress tracking
            total_minutes = int((end_dt - start_dt).total_seconds() / 60)
            if granularity == 'M1':
                # Forex market ~77% uptime (5.5 days/week, ~24hrs/day)
                estimated_candles = int(total_minutes * 0.77)
            else:
                estimated_candles = total_minutes  # Conservative estimate
            
            logger.info(f"📊 Estimated candles: {estimated_candles:,}")
            
            # Create output directory
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize CSV file with headers
            with open(output_path, 'w') as f:
                f.write("timestamp,open,high,low,close,volume\n")
            
            # Download data in batches
            current_start = start_dt
            batch_num = 0
            total_candles = 0
            
            while current_start < end_dt:
                batch_num += 1
                
                # Download batch
                batch_df = self._download_batch(
                    instrument=instrument,
                    granularity=granularity,
                    start_time=current_start,
                    end_time=end_dt,
                    batch_num=batch_num
                )
                
                if batch_df is None or len(batch_df) == 0:
                    logger.warning(f"⚠️  No data received for batch {batch_num}")
                    # Try moving forward by 1 day
                    current_start += timedelta(days=1)
                    continue
                
                # Append to CSV (memory efficient)
                batch_df.to_csv(output_path, mode='a', header=False, index=False)
                
                # Update progress
                total_candles += len(batch_df)
                progress = (total_candles / estimated_candles) * 100 if estimated_candles > 0 else 0
                
                logger.info(f"📈 Progress: {progress:.1f}% ({total_candles:,}/{estimated_candles:,} candles)")
                logger.info(f"   Last timestamp: {batch_df.iloc[-1]['timestamp']}")
                
                # Move to next batch (use last timestamp + 1 minute)
                last_time = pd.to_datetime(batch_df.iloc[-1]['timestamp'])
                if last_time.tzinfo is None:
                    last_time = last_time.replace(tzinfo=timezone.utc)
                
                current_start = last_time + timedelta(minutes=1)
                
                # Safety check - prevent infinite loops
                if batch_num > 1000:  # ~5M candles max
                    logger.warning("⚠️  Reached safety limit of 1000 batches")
                    break
                
                # Rate limiting
                time.sleep(self.RATE_LIMIT_DELAY)
            
            # Final validation
            final_df = pd.read_csv(output_path)
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            
            logger.info(f"✅ Download complete!")
            logger.info(f"📊 Final stats:")
            logger.info(f"   Total candles: {len(final_df):,}")
            logger.info(f"   File size: {file_size_mb:.2f} MB")
            logger.info(f"   Date range: {final_df.iloc[0]['timestamp']} to {final_df.iloc[-1]['timestamp']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return False
    
    def _download_batch(
        self,
        instrument: str,
        granularity: str,
        start_time: datetime,
        end_time: datetime,
        batch_num: int
    ) -> Optional[pd.DataFrame]:
        """Download a single batch of candles"""
        try:
            logger.info(f"🔄 Batch {batch_num}: Requesting data from {start_time}")
            
            # Prepare API parameters
            params = {
                "granularity": granularity,
                "fromTime": start_time.strftime('%Y-%m-%dT%H:%M:%S.000000000Z'),
                "count": self.MAX_CANDLES_PER_REQUEST,
                "includeFirst": False if batch_num > 1 else True
            }
            
            # Add end time if we're near the target end date
            remaining_time = end_time - start_time
            if remaining_time.total_seconds() < (7 * 24 * 60 * 60):  # Less than 1 week
                params["toTime"] = end_time.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
                params.pop("count", None)  # Remove count when using both from and to
            
            # Make API request
            response = self.api_context.instrument.candles(
                instrument=instrument,
                **params
            )
            
            if response.status != 200:
                logger.error(f"❌ API error: {response.status}, body: {response.body}")
                return None
            
            # Process response
            candles_data = response.body.get('candles', [])
            if not candles_data:
                return None
            
            # Convert to DataFrame
            processed_candles = []
            for candle in candles_data:
                if candle.complete:  # Only complete candles
                    processed_candles.append({
                        'timestamp': candle.time,
                        'open': float(candle.mid.o),
                        'high': float(candle.mid.h),
                        'low': float(candle.mid.l),
                        'close': float(candle.mid.c),
                        'volume': int(candle.volume)
                    })
            
            if not processed_candles:
                return None
            
            df = pd.DataFrame(processed_candles)
            
            # Data validation
            if len(df) > 0:
                logger.info(f"✅ Batch {batch_num}: {len(df):,} candles from {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Batch {batch_num} failed: {e}")
            return None
    
    @classmethod
    def from_env_file(cls, env_file: str = ".env", environment: str = 'live'):
        """Create downloader using credentials from .env file"""
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            
            api_key = os.getenv("OANDA_API_KEY")
            account_id = os.getenv("OANDA_ACCOUNT_ID")
            
            if not api_key:
                raise ValueError("OANDA_API_KEY not found in environment")
            if not account_id:
                raise ValueError("OANDA_ACCOUNT_ID not found in environment")
            
            return cls(api_key, account_id, environment)
            
        except Exception as e:
            logger.error(f"❌ Failed to load credentials: {e}")
            raise

def download_gbpjpy_dataset():
    """Download the specific GBPJPY dataset requested"""
    try:
        # Create downloader from .env file
        downloader = SWTOandaDownloader.from_env_file()
        
        # Define parameters
        instrument = "GBP_JPY"
        start_date = "2022-01-01"
        end_date = "2025-08-07"
        output_path = "../data/GBPJPY_M1_202201-202508.csv"
        
        # Download data
        success = downloader.download_historical_csv(
            instrument=instrument,
            start_date=start_date,
            end_date=end_date,
            output_path=output_path,
            granularity='M1'
        )
        
        if success:
            logger.info(f"🎉 Successfully created {output_path}")
            return True
        else:
            logger.error(f"❌ Failed to create {output_path}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Dataset download failed: {e}")
        return False

def main():
    """Main entry point with command line support"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SWT Oanda Data Downloader")
    parser.add_argument("--instrument", default="GBP_JPY", help="Forex instrument")
    parser.add_argument("--start-date", default="2022-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2025-08-07", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", help="Output CSV file path")
    parser.add_argument("--granularity", default="M1", help="Timeframe (M1, M5, H1)")
    parser.add_argument("--env-file", default=".env", help="Environment file path")
    parser.add_argument("--environment", default="live", choices=["live", "practice"], help="Oanda environment")
    
    args = parser.parse_args()
    
    # Default output path if not specified
    if not args.output:
        args.output = f"../data/{args.instrument}_{args.granularity}_{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}.csv"
    
    try:
        # Create downloader
        downloader = SWTOandaDownloader.from_env_file(args.env_file, args.environment)
        
        # Download data
        success = downloader.download_historical_csv(
            instrument=args.instrument,
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=args.output,
            granularity=args.granularity
        )
        
        if success:
            print(f"✅ Download completed: {args.output}")
            return 0
        else:
            print(f"❌ Download failed")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    # Check if we should run the specific GBPJPY download
    if len(sys.argv) == 1:
        print("🚀 Running default GBPJPY dataset download...")
        success = download_gbpjpy_dataset()
        sys.exit(0 if success else 1)
    else:
        sys.exit(main())