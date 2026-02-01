import asyncio
import time
import pytest
from app.llm_client import RateLimiter

@pytest.mark.asyncio
async def test_rate_limiter_throttling():
    # Use 0.2s interval for faster test execution
    interval = 0.2
    limiter = RateLimiter(interval=interval)
    
    start = time.monotonic()
    
    # First acquire should be almost immediate
    await limiter.acquire()
    t1 = time.monotonic()
    
    # Second acquire should delay
    await limiter.acquire()
    t2 = time.monotonic()
    
    # Third acquire should delay
    await limiter.acquire()
    t3 = time.monotonic()
    
    # Validation
    # First call is fast
    assert (t1 - start) < 0.1, f"First acquire took too long: {t1 - start}"
    
    # Subsequent calls should respect interval
    # We use a slight tolerance (0.01s) for system timer resolution
    assert (t2 - t1) >= (interval - 0.05), f"Interval 1 was too short: {t2 - t1}"
    assert (t3 - t2) >= (interval - 0.05), f"Interval 2 was too short: {t3 - t2}"
    
    # Total time for 3 calls (2 delays) should be around 2 * interval
    total_time = t3 - start
    assert total_time >= (2 * interval - 0.1), f"Total time too short: {total_time}"
