"""
Embedding 性能基准测试工具

用于对比优化前后的性能差异，测量实际的吞吐量和延迟。
"""

import asyncio
import time
from pathlib import Path
import sys
import json

# 添加项目路径
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from src.agentic_layer import vectorize_service


async def benchmark_serial(texts: list[str], batch_size: int = 256):
    """串行处理（优化前的方式）"""
    print(f"\n{'='*60}")
    print(f"Benchmark: Serial Processing (batch_size={batch_size})")
    print(f"{'='*60}")
    
    start_time = time.time()
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_idx = i // batch_size + 1
        
        batch_start = time.time()
        batch_embeddings = await vectorize_service.get_text_embeddings(batch_texts)
        batch_time = time.time() - batch_start
        
        all_embeddings.extend(batch_embeddings)
        print(f"  Batch {batch_idx}/{total_batches}: {len(batch_texts)} texts, {batch_time:.2f}s")
    
    elapsed_time = time.time() - start_time
    speed = len(texts) / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\n✅ Serial Processing Complete!")
    print(f"   - Total texts: {len(texts)}")
    print(f"   - Total embeddings: {len(all_embeddings)}")
    print(f"   - Time elapsed: {elapsed_time:.2f}s")
    print(f"   - Speed: {speed:.1f} texts/sec")
    print(f"   - Average batch time: {elapsed_time/total_batches:.2f}s")
    
    return {
        "method": "serial",
        "batch_size": batch_size,
        "total_texts": len(texts),
        "total_embeddings": len(all_embeddings),
        "elapsed_time": elapsed_time,
        "speed": speed,
        "avg_batch_time": elapsed_time / total_batches
    }


async def benchmark_parallel(texts: list[str], batch_size: int = 100, max_concurrent: int = 10):
    """并发处理（优化后的方式）"""
    print(f"\n{'='*60}")
    print(f"Benchmark: Parallel Processing (batch_size={batch_size}, max_concurrent={max_concurrent})")
    print(f"{'='*60}")
    
    start_time = time.time()
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    async def process_batch(batch_idx: int, batch_texts: list) -> tuple[int, list]:
        """处理单个批次"""
        batch_start = time.time()
        try:
            batch_embeddings = await vectorize_service.get_text_embeddings(batch_texts)
            batch_time = time.time() - batch_start
            print(f"  Batch {batch_idx + 1}/{total_batches}: {len(batch_texts)} texts, {batch_time:.2f}s")
            return (batch_idx, batch_embeddings)
        except Exception as e:
            print(f"  ❌ Batch {batch_idx + 1}/{total_batches} failed: {e}")
            return (batch_idx, [])
    
    # 创建所有批次任务
    tasks = []
    for i in range(0, len(texts), batch_size):
        batch_idx = i // batch_size
        batch_texts = texts[i : i + batch_size]
        task = process_batch(batch_idx, batch_texts)
        tasks.append(task)
    
    print(f"Submitting {len(tasks)} batches for concurrent processing...")
    
    # 分批提交任务（避免内存问题）
    batch_results = []
    completed = 0
    chunk_size = max_concurrent * 2
    
    for chunk_start in range(0, len(tasks), chunk_size):
        chunk_tasks = tasks[chunk_start : chunk_start + chunk_size]
        chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=False)
        batch_results.extend(chunk_results)
        
        completed += len(chunk_tasks)
        progress = (completed / len(tasks)) * 100
        print(f"  Progress: {completed}/{len(tasks)} batches ({progress:.1f}%)")
    
    # 按批次顺序重组结果
    all_embeddings = []
    for batch_idx, batch_embeddings in sorted(batch_results, key=lambda x: x[0]):
        all_embeddings.extend(batch_embeddings)
    
    elapsed_time = time.time() - start_time
    speed = len(texts) / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\n✅ Parallel Processing Complete!")
    print(f"   - Total texts: {len(texts)}")
    print(f"   - Total embeddings: {len(all_embeddings)}")
    print(f"   - Time elapsed: {elapsed_time:.2f}s")
    print(f"   - Speed: {speed:.1f} texts/sec")
    print(f"   - Average batch time: {elapsed_time/total_batches:.2f}s")
    
    return {
        "method": "parallel",
        "batch_size": batch_size,
        "max_concurrent": max_concurrent,
        "total_texts": len(texts),
        "total_embeddings": len(all_embeddings),
        "elapsed_time": elapsed_time,
        "speed": speed,
        "avg_batch_time": elapsed_time / total_batches
    }


def print_comparison(serial_result: dict, parallel_result: dict):
    """打印对比结果"""
    print(f"\n{'='*60}")
    print(f"Performance Comparison")
    print(f"{'='*60}")
    
    speedup = serial_result["speed"] / parallel_result["speed"] if parallel_result["speed"] > 0 else 0
    time_saved = serial_result["elapsed_time"] - parallel_result["elapsed_time"]
    time_saved_pct = (time_saved / serial_result["elapsed_time"]) * 100
    
    print(f"\n📊 Summary:")
    print(f"   Serial Processing:")
    print(f"      - Time: {serial_result['elapsed_time']:.2f}s")
    print(f"      - Speed: {serial_result['speed']:.1f} texts/sec")
    print(f"      - Batch size: {serial_result['batch_size']}")
    
    print(f"\n   Parallel Processing:")
    print(f"      - Time: {parallel_result['elapsed_time']:.2f}s")
    print(f"      - Speed: {parallel_result['speed']:.1f} texts/sec")
    print(f"      - Batch size: {parallel_result['batch_size']}")
    print(f"      - Max concurrent: {parallel_result['max_concurrent']}")
    
    print(f"\n🚀 Performance Improvement:")
    print(f"      - Speedup: {speedup:.2f}x")
    print(f"      - Time saved: {time_saved:.2f}s ({time_saved_pct:.1f}%)")
    
    if speedup >= 3:
        print(f"      - 🎉 Excellent! {speedup:.1f}x speedup achieved!")
    elif speedup >= 2:
        print(f"      - ✅ Good! {speedup:.1f}x speedup achieved!")
    else:
        print(f"      - ⚠️  Modest improvement. Consider tuning parameters.")


async def main():
    """主测试函数"""
    print("="*60)
    print("Embedding Performance Benchmark")
    print("="*60)
    
    # 加载测试数据（使用 conv_0 的数据）
    data_dir = Path(__file__).parent.parent / "results" / "locomo_evaluation_event_log3" / "memcells"
    test_file = data_dir / "memcell_list_conv_0.json"
    
    if not test_file.exists():
        print(f"Error: Test file not found: {test_file}")
        print("Please run stage1_memcells_extraction.py first.")
        return
    
    print(f"\nLoading test data from: {test_file.name}")
    with open(test_file, "r", encoding="utf-8") as f:
        docs = json.load(f)
    
    # 提取 atomic_facts
    texts_to_embed = []
    for doc in docs:
        if doc.get("event_log") and doc["event_log"].get("atomic_fact"):
            atomic_facts = doc["event_log"]["atomic_fact"]
            if isinstance(atomic_facts, list):
                for fact in atomic_facts:
                    if fact and isinstance(fact, str) and fact.strip():
                        texts_to_embed.append(fact)
    
    print(f"Loaded {len(texts_to_embed)} texts from {len(docs)} documents")
    
    # 为了加快测试，只使用前 200 个文本
    # 如果要测试完整数据集，注释掉这行
    texts_to_embed = texts_to_embed[:200]
    print(f"Using {len(texts_to_embed)} texts for benchmark")
    
    # 测试串行处理
    serial_result = await benchmark_serial(texts_to_embed, batch_size=256)
    
    # 等待一下，避免 API 限流
    print("\nWaiting 3 seconds before parallel benchmark...")
    await asyncio.sleep(3)
    
    # 测试并发处理
    parallel_result = await benchmark_parallel(texts_to_embed, batch_size=100, max_concurrent=10)
    
    # 打印对比结果
    print_comparison(serial_result, parallel_result)
    
    # 保存结果到文件
    results = {
        "serial": serial_result,
        "parallel": parallel_result,
        "speedup": serial_result["speed"] / parallel_result["speed"] if parallel_result["speed"] > 0 else 0
    }
    
    output_file = Path(__file__).parent / "benchmark_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Benchmark results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())

