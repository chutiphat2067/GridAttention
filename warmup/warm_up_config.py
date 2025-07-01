# warm_up_config.py
WARMUP_CONFIG = {
    'target_observations': {
        'learning_phase': 100000,    # แทนที่จะรอ 2000 trades
        'shadow_phase': 20000,        # แทนที่จะรอ 500 trades
        'active_phase': 10000         # แทนที่จะรอ 200 trades
    },
    
    'sampling_strategy': {
        'total_samples': 1000000,     # 1M จาก 36M
        'recent_weight': 0.7,         # 70% จากข้อมูลล่าสุด
        'volatility_weight': 0.2,     # 20% จากช่วง high volatility
        'regime_diversity_weight': 0.1 # 10% เพื่อความหลากหลาย
    },
    
    'processing_params': {
        'batch_size': 10000,
        'feature_extraction_interval': 5,  # ทุก 5 วินาที
        'progress_report_interval': 50000
    }
}