#!/usr/bin/env python3
"""
Cache Performance Monitor for MemMimic

Real-time monitoring of cache performance and optimization insights.
"""

import json
import logging
import sys
import time
from pathlib import Path

# Add MemMimic to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from memmimic.utils.caching import get_cache_statistics, clear_all_caches

def monitor_cache_performance(duration: int = 60, interval: int = 5):
    """
    Monitor cache performance for a specified duration.
    
    Args:
        duration: Total monitoring duration in seconds
        interval: Reporting interval in seconds
    """
    print(f"🔍 Starting cache monitoring for {duration}s (reporting every {interval}s)")
    print("=" * 70)
    
    start_time = time.time()
    while time.time() - start_time < duration:
        stats = get_cache_statistics()
        
        # Display cache statistics
        print(f"\n📊 Cache Statistics ({time.strftime('%H:%M:%S')})")
        print("-" * 50)
        
        for cache_name, cache_stats in stats.items():
            if cache_name == "total_cached_operations":
                continue
                
            print(f"\n{cache_name.upper()}:")
            print(f"  Size: {cache_stats['cache_size']}/{cache_stats['max_size']}")
            print(f"  Hit Rate: {cache_stats['hit_rate']:.1%}")
            print(f"  Hits: {cache_stats['hits']} | Misses: {cache_stats['misses']}")
            print(f"  Evictions: {cache_stats['evictions']}")
        
        print(f"\n🎯 Total Operations Cached: {stats['total_cached_operations']}")
        
        # Calculate overall performance
        total_requests = sum(
            cache['total_requests'] 
            for cache in stats.values() 
            if isinstance(cache, dict) and 'total_requests' in cache
        )
        
        total_hits = sum(
            cache['hits'] 
            for cache in stats.values() 
            if isinstance(cache, dict) and 'hits' in cache
        )
        
        overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0
        
        print(f"📈 Overall Hit Rate: {overall_hit_rate:.1%}")
        
        # Performance indicators
        if overall_hit_rate >= 0.8:
            print("✅ Excellent cache performance!")
        elif overall_hit_rate >= 0.6:
            print("⚡ Good cache performance")
        elif overall_hit_rate >= 0.4:
            print("⚠️  Moderate cache performance - consider tuning")
        else:
            print("🚨 Poor cache performance - optimization needed")
        
        time.sleep(interval)
    
    print(f"\n🏁 Monitoring completed after {duration}s")


def export_cache_report(filename: str = "cache_report.json"):
    """Export detailed cache statistics to JSON file."""
    stats = get_cache_statistics()
    
    report = {
        "timestamp": time.time(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cache_statistics": stats,
        "recommendations": []
    }
    
    # Generate recommendations
    for cache_name, cache_stats in stats.items():
        if isinstance(cache_stats, dict) and 'hit_rate' in cache_stats:
            hit_rate = cache_stats['hit_rate']
            
            if hit_rate < 0.6:
                report["recommendations"].append({
                    "cache": cache_name,
                    "issue": "Low hit rate",
                    "hit_rate": hit_rate,
                    "suggestion": "Consider increasing TTL or cache size"
                })
            
            if cache_stats['evictions'] > cache_stats['hits'] * 0.1:
                report["recommendations"].append({
                    "cache": cache_name,
                    "issue": "High eviction rate", 
                    "evictions": cache_stats['evictions'],
                    "suggestion": "Increase cache max_size"
                })
    
    # Save report
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Cache report exported to {filename}")
    return report


def clear_cache_interactive():
    """Interactively clear caches with confirmation."""
    stats = get_cache_statistics()
    
    print("🗑️  Current Cache Status:")
    for cache_name, cache_stats in stats.items():
        if isinstance(cache_stats, dict) and 'cache_size' in cache_stats:
            print(f"  {cache_name}: {cache_stats['cache_size']} items")
    
    response = input("\nClear all caches? (y/N): ").strip().lower()
    
    if response == 'y':
        clear_all_caches()
        print("✅ All caches cleared successfully")
    else:
        print("❌ Cache clear cancelled")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MemMimic Cache Monitor")
    parser.add_argument(
        "--monitor", "-m", 
        type=int, 
        default=0,
        help="Monitor cache for N seconds"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=5,
        help="Reporting interval in seconds"
    )
    parser.add_argument(
        "--export", "-e",
        type=str,
        help="Export cache report to file"
    )
    parser.add_argument(
        "--clear", "-c",
        action="store_true",
        help="Clear all caches"
    )
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Show current cache statistics"
    )
    
    args = parser.parse_args()
    
    if args.monitor > 0:
        monitor_cache_performance(args.monitor, args.interval)
    elif args.export:
        export_cache_report(args.export)
    elif args.clear:
        clear_cache_interactive()
    elif args.stats:
        stats = get_cache_statistics()
        print(json.dumps(stats, indent=2))
    else:
        print("MemMimic Cache Monitor")
        print("Use --help for usage options")
        print("\nQuick stats:")
        stats = get_cache_statistics()
        print(f"Total cached operations: {stats.get('total_cached_operations', 0)}")