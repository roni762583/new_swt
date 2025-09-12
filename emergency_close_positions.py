#!/usr/bin/env python3
"""
Emergency Position Closer - Manually close all open OANDA positions
"""

import os
import v20
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def close_all_positions():
    """Close all open positions in OANDA account"""
    
    # OANDA credentials
    api_key = os.getenv('OANDA_API_KEY')
    account_id = os.getenv('OANDA_ACCOUNT_ID')
    environment = os.getenv('OANDA_ENVIRONMENT', 'live')
    
    if not api_key or not account_id:
        print("❌ Missing OANDA credentials in .env file")
        return
    
    # Connect to OANDA
    api = v20.Context(
        hostname="api-fxtrade.oanda.com" if environment == "live" else "api-fxpractice.oanda.com",
        token=api_key
    )
    
    try:
        # Get all positions
        response = api.position.list(accountID=account_id)
        
        if response.status != 200:
            print(f"❌ Failed to get positions: {response.body}")
            return
        
        positions = response.body.get("positions", [])
        
        if not positions:
            print("✅ No open positions found")
            return
        
        print(f"🔍 Found {len(positions)} position(s)")
        
        for position in positions:
            instrument = position.instrument
            long_units = float(position.long.units) if position.long.units else 0
            short_units = float(position.short.units) if position.short.units else 0
            
            if long_units != 0 or short_units != 0:
                print(f"📊 Position: {instrument} - Long: {long_units}, Short: {short_units}")
                
                # Try market order to close position instead
                if short_units < 0:
                    # Close short position with a buy order
                    units_to_close = abs(int(short_units))
                    print(f"🔧 Closing short position with BUY order: {units_to_close} units")
                    
                    order_body = {
                        "order": {
                            "type": "MARKET",
                            "instrument": instrument,
                            "units": str(units_to_close),  # Positive units = BUY to close short
                            "timeInForce": "IOC"
                        }
                    }
                elif long_units > 0:
                    # Close long position with a sell order  
                    units_to_close = int(long_units)
                    print(f"🔧 Closing long position with SELL order: {units_to_close} units")
                    
                    order_body = {
                        "order": {
                            "type": "MARKET",
                            "instrument": instrument,
                            "units": str(-units_to_close),  # Negative units = SELL to close long
                            "timeInForce": "IOC"
                        }
                    }
                
                print(f"📤 Order body: {order_body}")
                close_response = api.order.market(account_id, **order_body["order"])
                
                if close_response.status == 200:
                    print(f"✅ Closed position: {instrument}")
                    
                    # Print close details
                    long_close = close_response.body.get("longOrderFillTransaction")
                    short_close = close_response.body.get("shortOrderFillTransaction")
                    
                    close_transaction = long_close or short_close
                    if close_transaction:
                        units_closed = abs(int(close_transaction.units))
                        close_price = float(close_transaction.price)
                        realized_pnl = float(close_transaction.pl)
                        print(f"   Units closed: {units_closed}")
                        print(f"   Close price: {close_price:.5f}")
                        print(f"   Realized P&L: ${realized_pnl:.2f}")
                else:
                    print(f"❌ Failed to close {instrument}: {close_response.body}")
        
        print("🎉 Emergency position close complete!")
        
    except Exception as e:
        print(f"❌ Emergency close failed: {e}")

if __name__ == "__main__":
    close_all_positions()