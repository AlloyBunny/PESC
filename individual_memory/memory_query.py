"""
简化的记忆数据库查询模块
用于在其他Python文件中快速使用记忆检索功能
"""

from individual_memory.memory_db_builder import MemoryVectorDB

# 全局配置 - 根据你的实际情况修改这些路径
BGE_MODEL_PATH = "models/bge-m3"
# DB_STORAGE_PATH = "./memory_chroma_db"
DB_STORAGE_PATH_ZH = "individual_memory/memory_chroma_db_zh"
DB_STORAGE_PATH_EN = "individual_memory/memory_chroma_db_en"
COLLECTION_NAME = "dialogue_memories"
# BGE模型部署的显卡索引（在CUDA_VISIBLE_DEVICES中的索引）
# 例如：如果设置MODEL_GPU_INDEX=1，且export CUDA_VISIBLE_DEVICES=2,3
# 那么模型将部署在物理显卡3上（CUDA_VISIBLE_DEVICES中的第1个索引）
MODEL_GPU_INDEX = 0  # 默认使用第0个可见显卡

# 全局数据库实例和线程锁
_memory_db = None
_lock = None

def get_memory_db(lang: str='zh'):
    """
    获取记忆数据库实例（线程安全的单例模式）
    第一次调用时会初始化数据库，后续调用直接返回已初始化的实例
    """
    global _memory_db, _lock
    
    # 延迟导入threading，避免循环导入
    import threading
    if _lock is None:
        _lock = threading.Lock()
    
    # 双重检查锁定模式确保线程安全
    if _memory_db is None:
        with _lock:
            if _memory_db is None:
                print("正在初始化MemoryVectorDB和BAAI/bge-m3模型...")
                if lang == 'zh':
                    DB_STORAGE_PATH = DB_STORAGE_PATH_ZH
                else:
                    DB_STORAGE_PATH = DB_STORAGE_PATH_EN
                _memory_db = MemoryVectorDB(
                    model_path=BGE_MODEL_PATH,
                    db_path=DB_STORAGE_PATH,
                    collection_name=COLLECTION_NAME,
                    gpu_index=MODEL_GPU_INDEX
                )
                print("MemoryVectorDB初始化完成！")
    return _memory_db

def query_memories(query_text: str, user_id: str, memory_type: str, n_results: int = 1, index_before: int = None, lang: str='zh'):
    """
    查询记忆的便捷函数
    
    Args:
        query_text (str): 查询文本
        n_results (int): 返回结果数量，默认1
        user_id (str): 用户ID，可选
        memory_type (str): 记忆类型 ('L1'、'L2' 或 'L3')，可选
        index_before (int): 索引值上限，只返回index严格小于此值的记录，可选
    
    Returns:
        List[Dict]: 包含相似度和记忆内容的字典列表
        每个字典包含:
        - similarity: 余弦相似度 (0-1之间，1表示最相似)
        - memory: 记忆内容文本
    """
    memory_db = get_memory_db(lang)
    results = memory_db.query_memories(query_text, n_results, user_id, memory_type, index_before)
    
    # 如果没有结果，返回空列表
    if not results or not results.get('ids') or not results['ids'][0]:
        return []
    
    # 提取结果并格式化
    distances = results['distances'][0]
    documents = results['documents'][0]
    
    formatted_results = []
    for i in range(len(documents)):
        formatted_results.append({
            'similarity': round(1 - distances[i], 4),  # 余弦距离转余弦相似度
            'memory': documents[i]
        })
    
    return formatted_results
