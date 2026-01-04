"""
Integration Test for Vectorize and Rerank Services with Real Configuration

Tests the embedding and reranking services using actual environment configuration.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path (for running from tests directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
from agentic_layer.vectorize_service import get_vectorize_service
from agentic_layer.rerank_service import get_rerank_service
from api_specs.memory_models import MemoryType
from agentic_layer.rerank_interface import RerankMemResponse

# Load environment variables
load_dotenv(project_root / ".env")


async def test_vectorize_service():
    """Test vectorization service with real configuration"""
    print("\n" + "=" * 80)
    print("🔹 Testing Vectorize Service")
    print("=" * 80)
    
    vectorize_service = get_vectorize_service()
    
    # Display configuration
    print(f"\n📋 Configuration:")
    print(f"   Provider: {os.getenv('VECTORIZE_PROVIDER', 'N/A')}")
    print(f"   Base URL: {os.getenv('VECTORIZE_BASE_URL', 'N/A')}")
    print(f"   Model: {os.getenv('VECTORIZE_MODEL', 'N/A')}")
    print(f"   Fallback Provider: {os.getenv('VECTORIZE_FALLBACK_PROVIDER', 'N/A')}")
    print(f"   Fallback Base URL: {os.getenv('VECTORIZE_FALLBACK_BASE_URL', 'N/A')}")
    print(f"   Dimensions: {os.getenv('VECTORIZE_DIMENSIONS', 'N/A')}")
    
    # Test queries
    test_texts = [
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language",
        "Deep learning uses neural networks for pattern recognition"
    ]
    
    print(f"\n🧪 Testing with {len(test_texts)} texts...")
    
    try:
        # Test single embedding
        print("\n1️⃣ Testing single embedding...")
        single_embedding = await vectorize_service.get_embedding(test_texts[0])
        print(f"   ✅ Single embedding shape: {single_embedding.shape}")
        print(f"   ✅ First 5 values: {single_embedding[:5]}")
        print(f"   ✅ Norm: {(single_embedding ** 2).sum() ** 0.5:.4f}")
        
        # Test batch embeddings
        print("\n2️⃣ Testing batch embeddings...")
        batch_embeddings = await vectorize_service.get_embeddings(test_texts)
        print(f"   ✅ Batch embeddings count: {len(batch_embeddings)}")
        for i, emb in enumerate(batch_embeddings):
            print(f"   ✅ Text {i+1} shape: {emb.shape}, norm: {(emb ** 2).sum() ** 0.5:.4f}")
        
        # Test query embedding
        print("\n3️⃣ Testing query embedding...")
        query_embedding = await vectorize_service.get_embedding(
            "What is machine learning?", 
            is_query=True
        )
        print(f"   ✅ Query embedding shape: {query_embedding.shape}")
        print(f"   ✅ Query norm: {(query_embedding ** 2).sum() ** 0.5:.4f}")
        
        # Calculate similarities
        print("\n4️⃣ Testing similarity calculation...")
        similarities = []
        for i, doc_emb in enumerate(batch_embeddings):
            # Cosine similarity
            similarity = (query_embedding * doc_emb).sum() / (
                ((query_embedding ** 2).sum() ** 0.5) * ((doc_emb ** 2).sum() ** 0.5)
            )
            similarities.append((i, similarity))
            print(f"   📊 Text {i+1} similarity: {similarity:.4f}")
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        print(f"\n   🏆 Most relevant: Text {similarities[0][0]+1} (score: {similarities[0][1]:.4f})")
        
        print("\n✅ Vectorize service test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Vectorize service test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await vectorize_service.close()


async def test_rerank_service():
    """Test reranking service with real configuration"""
    print("\n" + "=" * 80)
    print("🔹 Testing Rerank Service")
    print("=" * 80)
    
    rerank_service = get_rerank_service()
    
    # Display configuration
    print(f"\n📋 Configuration:")
    print(f"   Provider: {os.getenv('RERANK_PROVIDER', 'N/A')}")
    print(f"   Base URL: {os.getenv('RERANK_BASE_URL', 'N/A')}")
    print(f"   Model: {os.getenv('RERANK_MODEL', 'N/A')}")
    print(f"   Fallback Provider: {os.getenv('RERANK_FALLBACK_PROVIDER', 'N/A')}")
    print(f"   Fallback Base URL: {os.getenv('RERANK_FALLBACK_BASE_URL', 'N/A')}")
    
    # Test query and documents
    query = "What is machine learning and how does it work?"
    documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming. It uses algorithms to identify patterns and make predictions.",
        "Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data analysis, and automation.",
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers. It's particularly effective for image recognition and natural language processing.",
        "Data science combines statistics, programming, and domain expertise to extract insights from data. It's used across industries for decision-making.",
        "Neural networks are computational models inspired by the human brain. They consist of interconnected nodes that process information in layers."
    ]
    
    print(f"\n🧪 Testing with query and {len(documents)} documents...")
    print(f"   Query: '{query}'")
    
    try:
        # Test document reranking
        print("\n1️⃣ Testing document reranking...")
        reranked_docs = await rerank_service.rerank_documents(query, documents)
        
        print(f"   ✅ Reranked {len(reranked_docs)} documents")
        print(f"\n   📊 Reranking results (sorted by relevance):")
        for i, doc in enumerate(reranked_docs[:5]):  # Show top 5
            doc_text = doc['document'][:80] + "..." if len(doc['document']) > 80 else doc['document']
            print(f"   {i+1}. Score: {doc['relevance_score']:.4f}")
            print(f"      Text: {doc_text}")
            print()
        
        # Verify ranking
        print("2️⃣ Verifying ranking quality...")
        top_doc = reranked_docs[0]['document']
        if "machine learning" in top_doc.lower():
            print("   ✅ Top result contains 'machine learning' - ranking is good!")
        else:
            print(f"   ⚠️  Top result doesn't mention machine learning explicitly")
            print(f"      (This might still be correct if using semantic similarity)")
        
        # Test with top_k
        print("\n3️⃣ Testing with top_k=3...")
        top_3 = await rerank_service.rerank_documents(query, documents, top_k=3)
        print(f"   ✅ Retrieved top {len(top_3)} documents")
        for i, doc in enumerate(top_3):
            print(f"   {i+1}. Score: {doc['relevance_score']:.4f}")
        
        print("\n✅ Rerank service test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Rerank service test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await rerank_service.close()


async def test_retrieval_pipeline():
    """Test complete retrieval pipeline: embed + rerank"""
    print("\n" + "=" * 80)
    print("🔹 Testing Complete Retrieval Pipeline")
    print("=" * 80)
    
    vectorize_service = get_vectorize_service()
    rerank_service = get_rerank_service()
    
    # Test data
    query = "How does deep learning work?"
    documents = [
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers to progressively extract higher-level features from raw input.",
        "Python programming language is known for its simplicity and readability, making it popular for beginners and experts alike.",
        "Machine learning algorithms can be supervised, unsupervised, or semi-supervised depending on the type of training data available.",
        "Neural networks consist of layers of interconnected nodes that process and transform information, mimicking the structure of the human brain.",
        "Data preprocessing is an essential step in machine learning that involves cleaning, transforming, and organizing data for analysis.",
    ]
    
    print(f"\n🧪 Testing retrieval pipeline...")
    print(f"   Query: '{query}'")
    print(f"   Documents: {len(documents)}")
    
    try:
        # Step 1: Generate embeddings
        print("\n📍 Step 1: Generate embeddings...")
        query_emb = await vectorize_service.get_embedding(query, is_query=True)
        doc_embs = await vectorize_service.get_embeddings(documents)
        print(f"   ✅ Query embedding: shape={query_emb.shape}")
        print(f"   ✅ Document embeddings: {len(doc_embs)} vectors")
        
        # Step 2: Calculate initial similarity scores
        print("\n📍 Step 2: Calculate similarity scores...")
        scores = []
        for i, doc_emb in enumerate(doc_embs):
            similarity = (query_emb * doc_emb).sum() / (
                ((query_emb ** 2).sum() ** 0.5) * ((doc_emb ** 2).sum() ** 0.5)
            )
            scores.append((i, similarity))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        print("   📊 Initial ranking (by embedding similarity):")
        for rank, (idx, score) in enumerate(scores[:3]):
            doc_preview = documents[idx][:60] + "..."
            print(f"   {rank+1}. Doc {idx+1}: {score:.4f} - {doc_preview}")
        
        # Step 3: Rerank
        print("\n📍 Step 3: Rerank with reranker...")
        reranked = await rerank_service.rerank_documents(query, documents, top_k=3)
        print("   📊 Final ranking (after reranking):")
        for rank, doc in enumerate(reranked):
            doc_preview = doc['document'][:60] + "..."
            print(f"   {rank+1}. Score: {doc['relevance_score']:.4f} - {doc_preview}")
        
        # Compare rankings
        print("\n📍 Step 4: Compare rankings...")
        initial_top_idx = scores[0][0]
        reranked_top_text = reranked[0]['document']
        
        if documents[initial_top_idx] == reranked_top_text:
            print("   ✅ Rankings agree - top result is the same")
        else:
            print("   ℹ️  Rankings differ - reranker provided different ordering")
            print("      This is normal as reranker uses more sophisticated cross-attention")
        
        print("\n✅ Complete pipeline test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await vectorize_service.close()
        await rerank_service.close()


async def test_compare_vllm_deepinfra_rerank():
    """Compare vLLM and DeepInfra rerank results for the same query"""
    print("\n" + "=" * 80)
    print("🔹 Comparing vLLM vs DeepInfra Rerank Results")
    print("=" * 80)
    
    # Import service classes directly
    from agentic_layer.rerank_vllm import VllmRerankService, VllmRerankConfig
    from agentic_layer.rerank_deepinfra import DeepInfraRerankService, DeepInfraRerankConfig
    
    # Test query and documents
    query = "What is machine learning and how does it work?"
    documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.",
        "Python is a high-level programming language known for its simplicity and readability.",
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
        "Data science combines statistics, programming, and domain expertise to extract insights from data.",
        "Neural networks are computational models inspired by the human brain."
    ]
    
    print(f"\n🧪 Testing with same query and {len(documents)} documents...")
    print(f"   Query: '{query}'")
    
    try:
        # Create vLLM service
        vllm_config = VllmRerankConfig(
            api_key=os.getenv("RERANK_API_KEY", "EMPTY"),
            base_url=os.getenv("RERANK_BASE_URL", "http://localhost:12000/v1/rerank"),
            model=os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-4B"),
        )
        vllm_service = VllmRerankService(vllm_config)
        
        # Create DeepInfra service
        deepinfra_config = DeepInfraRerankConfig(
            api_key=os.getenv("RERANK_FALLBACK_API_KEY", ""),
            base_url=os.getenv("RERANK_FALLBACK_BASE_URL", "https://api.deepinfra.com/v1/inference"),
            model=os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-4B"),
        )
        deepinfra_service = DeepInfraRerankService(deepinfra_config)
        
        print(f"\n📋 Service Configuration:")
        print(f"   vLLM:      {vllm_config.base_url}")
        print(f"   DeepInfra: {deepinfra_config.base_url}")
        print(f"   Model:     {vllm_config.model}")
        
        # Test vLLM
        print("\n1️⃣ Testing vLLM reranking...")
        vllm_results = await vllm_service.rerank_documents(query, documents)
        print(f"   ✅ vLLM returned {len(vllm_results)} results")
        
        # Test DeepInfra
        print("\n2️⃣ Testing DeepInfra reranking...")
        deepinfra_results = await deepinfra_service._rerank_all_hits(
            query,
            [{"_source": {"episode": doc}, "memory_type": "episodic_memory"} for doc in documents]
        )
        print(f"   ✅ DeepInfra returned {len(deepinfra_results)} results")
        
        # Compare results
        print("\n3️⃣ Comparing results...")
        print("\n   📊 vLLM Rankings:")
        for i, doc in enumerate(vllm_results[:5]):
            doc_text = doc['document'][:60] + "..."
            print(f"   {i+1}. Score: {doc['relevance_score']:.4f} | {doc_text}")
        
        print("\n   📊 DeepInfra Rankings:")
        for i, hit in enumerate(deepinfra_results[:5]):
            doc_text = hit.get('_source', {}).get('episode', '')[:60] + "..."
            score = hit.get('score', 0.0)
            print(f"   {i+1}. Score: {score:.4f} | {doc_text}")
        
        # Calculate ranking correlation
        print("\n4️⃣ Analyzing ranking consistency...")
        
        # Get top document from each service
        vllm_top_doc = vllm_results[0]['document']
        deepinfra_top_doc = deepinfra_results[0].get('_source', {}).get('episode', '')
        
        if vllm_top_doc == deepinfra_top_doc:
            print("   ✅ Top ranked document is the SAME across both services")
        else:
            print("   ℹ️  Top ranked documents DIFFER between services")
            print(f"      This is expected as different implementations may have slight variations")
        
        # Calculate score correlation for top 3
        print("\n   📈 Score comparison (Top 3):")
        for i in range(min(3, len(vllm_results), len(deepinfra_results))):
            vllm_score = vllm_results[i]['relevance_score']
            deepinfra_score = deepinfra_results[i].get('score', 0.0)
            diff = abs(vllm_score - deepinfra_score)
            print(f"   Doc {i+1}: vLLM={vllm_score:.4f}, DeepInfra={deepinfra_score:.4f}, diff={diff:.4f}")
        
        # Check if rankings are similar (allowing for small variations)
        print("\n5️⃣ Ranking similarity analysis...")
        
        # Extract document indices from results
        vllm_indices = [documents.index(doc['document']) for doc in vllm_results]
        deepinfra_indices = []
        for hit in deepinfra_results:
            doc_text = hit.get('_source', {}).get('episode', '')
            if doc_text in documents:
                deepinfra_indices.append(documents.index(doc_text))
        
        # Compare top 3 rankings
        vllm_top3 = set(vllm_indices[:3])
        deepinfra_top3 = set(deepinfra_indices[:3])
        overlap = vllm_top3.intersection(deepinfra_top3)
        overlap_rate = len(overlap) / 3 if len(vllm_top3) >= 3 and len(deepinfra_top3) >= 3 else 0
        
        print(f"   Top-3 overlap: {len(overlap)}/3 documents ({overlap_rate*100:.0f}%)")
        
        if overlap_rate >= 0.67:  # At least 2 out of 3 match
            print("   ✅ Rankings are highly consistent between services")
        else:
            print("   ⚠️  Rankings show significant differences")
            print("      This may be due to model version differences or API variations")
        
        print("\n✅ Comparison test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Comparison test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await vllm_service.close()
        await deepinfra_service.close()


async def main():
    """Run all integration tests"""
    print("\n" + "🚀 " * 20)
    print("  Integration Tests for Vectorize & Rerank Services")
    print("🚀 " * 20)
    
    results = []
    
    # Test vectorize service
    result_vectorize = await test_vectorize_service()
    results.append(("Vectorize Service", result_vectorize))
    
    # Test rerank service
    result_rerank = await test_rerank_service()
    results.append(("Rerank Service", result_rerank))
    
    # Test complete pipeline
    result_pipeline = await test_retrieval_pipeline()
    results.append(("Complete Pipeline", result_pipeline))
    
    # Test comparison between vLLM and DeepInfra
    result_comparison = await test_compare_vllm_deepinfra_rerank()
    results.append(("vLLM vs DeepInfra Comparison", result_comparison))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Test Summary")
    print("=" * 80)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name:40s} {status}")
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    
    print(f"\n   Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n   🎉 All tests PASSED!")
        return 0
    else:
        print(f"\n   ⚠️  {total - passed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

