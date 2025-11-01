import os
from dotenv import load_dotenv

load_dotenv()


class ExperimentConfig:
    experiment_name: str = "locomo_evaluation"
    mode: str = "cot"
    datase_path: str = "data/locomo10.json"
    use_emb: bool = True
    use_reranker: bool = True  # 启用 Reranker
    use_agentic_retrieval: bool = True
    use_multi_query: bool = True  #  启用多查询生成
    num_conv: int = 10
    
    #  检索配置
    use_hybrid_search: bool = True  # 是否使用混合检索（Embedding + BM25 + RRF）
    emb_recall_top_n: int = 40      # Embedding/混合检索召回数量
    reranker_top_n: int = 20        # Reranker 重排序返回数量
    
    # 混合检索参数（仅在 use_hybrid_search=True 时生效）
    hybrid_emb_candidates: int = 50   # Embedding 候选数量
    hybrid_bm25_candidates: int = 50  # BM25 候选数量
    hybrid_rrf_k: int = 40             # RRF 参数 k
    
    #  多查询检索参数（仅在 use_multi_query=True 时生效）
    multi_query_num: int = 3           # 期望生成的查询数量
    multi_query_top_n: int = 50        # 每个查询召回的文档数
    
    # Reranker 优化参数（高性能配置）
    reranker_batch_size: int = 20      # Reranker 批次大小
    reranker_max_retries: int = 3      # 每个批次的最大重试次数
    reranker_retry_delay: float = 0.8  # 重试间隔，指数退避
    reranker_timeout: float = 60.0     # 单个批次超时时间
    reranker_fallback_threshold: float = 0.3  # 成功率低于此值时降级到原始排序
    reranker_concurrent_batches: int = 5  #  增加并发：5 个批次并发
    
    reranker_instruction: str = (
    "Determine if the passage contains specific facts, entities (names, dates, locations), "
    "or details that directly answer the question.")
    
    llm_service: str = "openai"  # openai, gemini, vllm
    # experiment_name: str = "locomo_evaluation_nemori"
    llm_config: dict = {
        "openai": {
            "llm_provider": "openai",
            "model": "openai/gpt-4.1-mini",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": os.getenv("LLM_API_KEY"),
            "temperature": 0.3,
            "max_tokens": 16384,
        },
        "vllm": {
            "llm_provider": "openai",
            "model": "Qwen3-30B",
            "base_url": "http://0.0.0.0:8000/v1",
            "api_key": "123",
            "temperature": 0,
            "max_tokens": 20000,
        },
    }
    max_retries: int = 5
    max_concurrent_requests: int = 10