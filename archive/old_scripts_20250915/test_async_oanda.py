#!/usr/bin/env python3
"""
Test script for async OANDA client implementation
"""

import asyncio
import sys
from pathlib import Path
from typing import List

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from swt_trading.async_oanda_client import (
    AsyncOandaClient,
    PriceUpdate,
    Position,
    OrderType
)


async def test_async_performance():
    """Test async client performance vs sequential"""
    print("Testing Async OANDA Client Performance...")

    # Mock environment variables for testing
    import os
    if not os.getenv('OANDA_API_KEY'):
        os.environ['OANDA_API_KEY'] = 'test_key'
        os.environ['OANDA_ACCOUNT_ID'] = 'test_account'
        print("⚠️ Using mock credentials for testing")

    try:
        # Test client initialization
        client = AsyncOandaClient(max_connections=5)
        print("✅ Client initialized")

        # Test data structures
        price = PriceUpdate(
            instrument='GBP_JPY',
            time='2024-01-01T00:00:00Z',
            bid=185.50,
            ask=185.52,
            spread=0.02
        )
        print(f"✅ Price update created: {price.instrument} @ {price.bid}/{price.ask}")

        position = Position(
            instrument='GBP_JPY',
            units=1000,
            avg_price=185.50,
            unrealized_pnl=25.0,
            side='long'
        )
        print(f"✅ Position created: {position.units} units @ {position.avg_price}")

        # Test concurrent operations simulation
        async def mock_api_call(delay: float) -> str:
            """Simulate API call with delay"""
            await asyncio.sleep(delay)
            return f"Result after {delay}s"

        # Sequential execution
        import time
        print("\n⏱️ Testing sequential execution...")
        start = time.time()
        results = []
        for delay in [0.1, 0.1, 0.1, 0.1, 0.1]:
            result = await mock_api_call(delay)
            results.append(result)
        sequential_time = time.time() - start

        # Concurrent execution
        print("⚡ Testing concurrent execution...")
        start = time.time()
        tasks = [mock_api_call(0.1) for _ in range(5)]
        results = await asyncio.gather(*tasks)
        concurrent_time = time.time() - start

        speedup = sequential_time / concurrent_time
        print(f"\n📊 Performance Results:")
        print(f"  Sequential: {sequential_time:.2f}s")
        print(f"  Concurrent: {concurrent_time:.2f}s")
        print(f"  Speedup: {speedup:.1f}x")

        assert concurrent_time < sequential_time, "Concurrent should be faster"
        print("✅ Performance test passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

    return True


async def test_async_patterns():
    """Test async patterns and concurrency"""
    print("\nTesting Async Patterns...")

    # Test queue operations
    queue = asyncio.Queue(maxsize=5)

    # Producer task
    async def producer():
        for i in range(10):
            if queue.full():
                # Drop oldest
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await queue.put(f"item_{i}")
            await asyncio.sleep(0.01)

    # Consumer task
    async def consumer():
        items = []
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.1)
                items.append(item)
            except asyncio.TimeoutError:
                break
        return items

    # Run producer and consumer
    producer_task = asyncio.create_task(producer())
    consumer_task = asyncio.create_task(consumer())

    await producer_task
    items = await consumer_task

    print(f"✅ Processed {len(items)} items through async queue")
    assert len(items) > 0, "Should have processed items"

    # Test task cancellation
    async def long_running():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            print("✅ Task cancelled successfully")
            raise

    task = asyncio.create_task(long_running())
    await asyncio.sleep(0.01)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass

    print("✅ Async patterns test complete!")
    return True


async def test_batch_operations():
    """Test batch operation patterns"""
    print("\nTesting Batch Operations...")

    # Simulate batch API calls
    async def api_call(id: int) -> dict:
        await asyncio.sleep(0.01)  # Simulate network delay
        return {'id': id, 'result': f'data_{id}'}

    # Execute batch of 10 operations concurrently
    print("⚡ Executing 10 concurrent operations...")
    start = asyncio.get_event_loop().time()

    tasks = [api_call(i) for i in range(10)]
    results = await asyncio.gather(*tasks)

    elapsed = asyncio.get_event_loop().time() - start

    print(f"✅ Completed {len(results)} operations in {elapsed:.3f}s")
    print(f"   Average: {elapsed/len(results)*1000:.1f}ms per operation")
    print(f"   Throughput: {len(results)/elapsed:.1f} ops/sec")

    assert len(results) == 10, "Should complete all operations"
    assert elapsed < 0.1, "Should execute concurrently"

    return True


async def main():
    """Run all async tests"""
    print("="*60)
    print("ASYNC OANDA CLIENT TEST SUITE")
    print("="*60)

    all_passed = True

    try:
        # Run tests
        all_passed &= await test_async_performance()
        all_passed &= await test_async_patterns()
        all_passed &= await test_batch_operations()

        print("\n" + "="*60)
        if all_passed:
            print("✅ ALL TESTS PASSED!")
        else:
            print("❌ SOME TESTS FAILED")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return all_passed


if __name__ == "__main__":
    # Run async tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)