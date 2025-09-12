#!/usr/bin/env python3
"""
Download GBPJPY data for Episode 13475 validation
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent to path
sys.path.append(str(Path(__file__).parent))

from scripts.download_oanda_data import SWTOandaDownloader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    # Get credentials from environment
    api_key = os.getenv('OANDA_API_KEY')
    account_id = os.getenv('OANDA_ACCOUNT_ID')
    environment = os.getenv('OANDA_ENVIRONMENT', 'live')
    
    if not api_key or not account_id:
        print("❌ Missing OANDA credentials in .env file")
        return 1
    
    print(f"🔑 Using OANDA {environment} environment")
    print(f"📊 Account: {account_id}")
    
    # Initialize downloader
    downloader = SWTOandaDownloader(api_key, account_id, environment)
    
    # Download last 3 years of data (2022-01 to 2025-08)
    # This matches the expected filename GBPJPY_M1_202201-202508.csv
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2025, 8, 31)
    
    output_path = "data/GBPJPY_M1_202201-202508.csv"
    
    print(f"\n📥 Downloading GBPJPY M1 data")
    print(f"   Start: {start_date.strftime('%Y-%m-%d')}")
    print(f"   End: {end_date.strftime('%Y-%m-%d')}")
    print(f"   Output: {output_path}")
    
    try:
        success = downloader.download_historical_csv(
            instrument='GBP_JPY',
            start_date=start_date,
            end_date=end_date,
            output_path=output_path,
            granularity='M1'
        )
        
        if success:
            print(f"\n✅ Successfully downloaded GBPJPY data to {output_path}")
            # Check file size
            file_size = Path(output_path).stat().st_size / (1024 * 1024)
            print(f"   File size: {file_size:.2f} MB")
            return 0
        else:
            print("\n❌ Failed to download data")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error downloading data: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())