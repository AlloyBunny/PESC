import json
import os
import logging
from typing import List, Dict, Optional, Any
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MemoryVectorDB:
    """
    一个用于管理个性化对话记忆的向量数据库。
    
    该类使用 BAAI/bge-m3 作为嵌入模型，ChromaDB 作为向量存储。
    它支持添加 L1 和 L2 类型的记忆，并提供强大的查询功能。
    """

    def __init__(self, model_path: str, db_path: str = "./chroma_db", collection_name: str = "dialogue_memories", gpu_index: int = 0):
        """
        初始化向量数据库。

        Args:
            model_path (str): bge-m3 模型所在的本地路径。
            db_path (str): ChromaDB 数据库文件存储的路径。
            collection_name (str): 数据库中的集合名称。
            gpu_index (int): 在CUDA_VISIBLE_DEVICES中的显卡索引，默认为0。
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
            
        logging.info(f"正在初始化 MemoryVectorDB...")
        self.db_path = db_path
        self.collection_name = collection_name
        
        # 1. 加载嵌入模型
        logging.info(f"从路径 '{model_path}' 加载 BAAI/bge-m3 模型...")
        
        # 确定设备
        device = self._get_device(gpu_index)
        logging.info(f"将模型部署到设备: {device}")
        
        # 使用 SentenceTransformerEmbeddingFunction 可以让 ChromaDB 内部处理嵌入过程，更高效
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_path,
            device=device
        )
        logging.info(f"模型已加载，运行在设备: {self.embedding_function.device}")

        # 2. 初始化 ChromaDB 客户端
        # PersistentClient 会将数据持久化到磁盘
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # 3. 获取或创建集合
        logging.info(f"正在获取或创建集合: '{self.collection_name}'")
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}  # bge 模型推荐使用余弦相似度
        )
        logging.info("初始化完成。")

    def _get_device(self, gpu_index: int) -> str:
        """
        根据gpu_index和CUDA_VISIBLE_DEVICES环境变量确定设备。
        
        Args:
            gpu_index (int): 在CUDA_VISIBLE_DEVICES中的显卡索引
            
        Returns:
            str: 设备字符串，如 "cuda:0", "cuda:1" 或 "cpu"
        """
        import torch
        
        # 检查是否有CUDA可用
        if not torch.cuda.is_available():
            logging.info("CUDA不可用，使用CPU")
            return "cpu"
        
        # 获取CUDA_VISIBLE_DEVICES环境变量
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        
        if not cuda_visible_devices:
            # 如果没有设置CUDA_VISIBLE_DEVICES，直接使用gpu_index
            if gpu_index < torch.cuda.device_count():
                device = f"cuda:{gpu_index}"
                logging.info(f"未设置CUDA_VISIBLE_DEVICES，使用显卡 {gpu_index}")
            else:
                logging.warning(f"请求的显卡索引 {gpu_index} 超出可用范围，使用 cuda:0")
                device = "cuda:0"
        else:
            # 解析CUDA_VISIBLE_DEVICES
            visible_devices = [int(x.strip()) for x in cuda_visible_devices.split(',') if x.strip()]
            
            if gpu_index < len(visible_devices):
                # 在可见设备范围内，使用cuda:gpu_index
                device = f"cuda:{gpu_index}"
                physical_gpu = visible_devices[gpu_index]
                logging.info(f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}，使用索引{gpu_index}对应物理显卡{physical_gpu}")
            else:
                # 超出可见设备范围，使用第一个可见设备
                device = "cuda:0"
                physical_gpu = visible_devices[0]
                logging.warning(f"请求的显卡索引 {gpu_index} 超出可见设备范围，使用第一个可见设备 {physical_gpu}")
        
        return device

    def _load_json_data(self, file_path: str) -> List[Dict[str, Any]]:
        """一个安全的加载JSON文件的辅助函数。"""
        if not os.path.exists(file_path):
            logging.error(f"JSON 文件未找到: {file_path}")
            return []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"读取或解析文件 {file_path} 时出错: {e}")
            return []

    def add_memories_from_file(self, file_path: str, memory_type: str):
        """
        从 JSON 文件中读取记忆并将其添加到数据库中。

        Args:
            file_path (str): L1_memory.json、L2_memory.json 或 L3_memory.json 的路径。
            memory_type (str): 记忆类型，应为 'L1'、'L2' 或 'L3'。
        """
        if memory_type not in ['L1', 'L2', 'L3']:
            raise ValueError("memory_type 必须是 'L1'、'L2' 或 'L3'")

        logging.info(f"开始处理文件: {file_path}，类型: {memory_type}")
        
        data = self._load_json_data(file_path)
        if not data:
            logging.warning(f"文件 {file_path} 中没有数据可添加。")
            return

        documents, metadatas, ids = [], [], []
        
        memory_key = f"{memory_type}_memory" # e.g., 'L1_memory' or 'L2_memory'

        for item in data:
            if memory_key not in item or not item[memory_key]:
                logging.warning(f"跳过一个无效项目（缺少 '{memory_key}' 字段）: {item.get('memory_id')}")
                continue

            # 要被向量化的文本内容
            documents.append(item[memory_key])
            
            # 使用 memory_id 作为 ChromaDB 中的唯一ID
            ids.append(item['memory_id'])

            # # 元数据包含除文本外的所有信息，并额外添加 memory_type
            # metadata = item.copy()
            # metadata['memory_type'] = memory_type
            # del metadata[memory_key] # 从元数据中移除已被用作document的字段
            # metadatas.append(metadata)
            metadatas.append({
                'id': item['id'],  # 用户ID
                'memory_type': memory_type, # 记忆类型 L1 或 L2
                'index': int(item['index'])  # 索引值，确保为int类型
            })

        if not documents:
            logging.info("没有新的有效记忆可添加。")
            return

        # 使用 upsert 批量添加数据，如果ID已存在则会更新
        logging.info(f"正在向数据库中添加 {len(documents)} 条 {memory_type} 记忆...")
        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logging.info(f"成功添加 {len(documents)} 条 {memory_type} 记忆。")

    def query_memories(self, query_text: str, n_results: int = 5, user_id: Optional[str] = None, memory_type: Optional[str] = None, index_before: Optional[int] = None) -> Dict[str, List[Any]]:
        """
        查询与给定文本最相关的记忆。

        Args:
            query_text (str): 用于查询的文本。
            n_results (int): 希望返回的结果数量。
            user_id (Optional[str]): (可选) 用于过滤特定用户的ID。
            memory_type (Optional[str]): (可选) 用于过滤特定记忆类型 ('L1' 或 'L2')。
            index_before (Optional[int]): (可选) 用于过滤index值小于此值的记录。

        Returns:
            Dict[str, List[Any]]: 包含查询结果的字典，格式与chromadb返回的类似。
        """
        conditions = []
        if user_id:
            conditions.append({"id": user_id})
        if memory_type:
            if memory_type not in ['L1', 'L2', 'L3']:
                raise ValueError("memory_type must be 'L1', 'L2' or 'L3'")
            conditions.append({"memory_type": memory_type})
        if index_before is not None:
            conditions.append({"index": {"$lt": index_before}})
        
        # 根据条件数量构造最终的 where_filter
        where_filter = {}
        if len(conditions) > 1:
            where_filter = {"$and": conditions}
        elif len(conditions) == 1:
            where_filter = conditions[0]
            
        #####################################################################
        # 【2025.9.10】打印日志太多，去掉
        # logging.info(f"执行查询: '{query_text[:50]}...'")
        # logging.info(f"过滤器: {where_filter}, 返回结果数: {n_results}")

        if not where_filter:
            # 如果没有过滤器，where参数需要为None，而不是空字典
             results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
        else:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_filter
            )
        
        return results

def print_results(results: Dict[str, List[Any]]):
    """一个格式化打印查询结果的辅助函数。"""
    if not results or not results.get('ids') or not results['ids'][0]:
        print("未找到相关结果。")
        return
        
    print("\n--- 查询结果 ---")
    ids = results['ids'][0]
    distances = results['distances'][0]
    metadatas = results['metadatas'][0]
    documents = results['documents'][0]

    for i, doc_id in enumerate(ids):
        print(f"[{i+1}] ID: {doc_id} (相似度得分: {1 - distances[i]:.4f})") # 余弦距离转为相似度
        print(f"    类型: {metadatas[i].get('memory_type')}")
        print(f"    用户ID: {metadatas[i].get('id')}")
        print(f"    索引: {metadatas[i].get('index')}")
        print(f"    内容: {documents[i][:200]}...") # 打印前200个字符预览
        print("-" * 20)

# --- 主执行逻辑 ---
if __name__ == "__main__":
    # --- 配置参数 ---
    BGE_MODEL_PATH = "models/bge-m3" # 【重要】请确保此路径正确
    DB_STORAGE_PATH = "./memory_chroma_db_en"
    COLLECTION_NAME = "dialogue_memories"
    L1_MEMORY_FILE = "./en/L1_memory.json"
    L2_MEMORY_FILE = "./en/L2_memory.json"
    L3_MEMORY_FILE = "./en/L3_memory.json"

    try:
        # 1. 初始化数据库
        memory_db = MemoryVectorDB(
            model_path=BGE_MODEL_PATH,
            db_path=DB_STORAGE_PATH,
            collection_name=COLLECTION_NAME
        )
        
        # 2. 添加数据 (可以加上判断，避免重复添加)
        if memory_db.collection.count() == 0:
            print("\n数据库为空，开始添加记忆...")
            memory_db.add_memories_from_file(L1_MEMORY_FILE, 'L1')
            memory_db.add_memories_from_file(L2_MEMORY_FILE, 'L2')
            memory_db.add_memories_from_file(L3_MEMORY_FILE, 'L3')
            print(f"\n数据添加完毕。数据库中现在有 {memory_db.collection.count()} 条记录。")
        else:
            print(f"\n数据库已存在，包含 {memory_db.collection.count()} 条记录。跳过添加步骤。")

        # 3. 执行查询示例
        print("\n" + "="*50)
        print("示例1: 通用查询，寻找关于'内心空虚'的记忆")
        query1 = "感觉内心很空虚，做什么都提不起劲，怎么办？"
        results1 = memory_db.query_memories(query1, n_results=3)
        print_results(results1)

        print("\n" + "="*50)
        user_id_to_query = "17f12328-febc-41d0-be16-4a35b2738e2c"
        print(f"示例2: 针对特定用户 '{user_id_to_query}' 的查询")
        query2 = "这个人的性格是怎样的？"
        results2 = memory_db.query_memories(query2, n_results=3, user_id=user_id_to_query)
        print_results(results2)

        print("\n" + "="*50)
        print(f"示例3: 针对用户 '{user_id_to_query}'，只查询L1类型的具体对话记忆")
        query3 = "聊聊关于找点刺激的事情"
        results3 = memory_db.query_memories(query3, n_results=1, user_id=user_id_to_query, memory_type='L1')
        print_results(results3)

        print("\n" + "="*50)
        print(f"示例4: 针对用户 '{user_id_to_query}'，只查询L2类型的总结性记忆")
        query4 = "给我一份关于他的总体性格和偏好总结"
        results4 = memory_db.query_memories(query4, n_results=1, user_id=user_id_to_query, memory_type='L2')
        print_results(results4)
        
        print("\n" + "="*50)
        print(f"示例5: 针对用户 '{user_id_to_query}'，只查询L3类型的深度总结记忆")
        query5 = "这个人的核心性格特质和互动指南"
        results5 = memory_db.query_memories(query5, n_results=1, user_id=user_id_to_query, memory_type='L3')
        print_results(results5)

    except FileNotFoundError as e:
        logging.error(f"发生致命错误: {e}")
        logging.error("请确保 BAAI/bge-m3 模型的路径是正确的，并且您有权访问它。")
    except Exception as e:
        logging.error(f"程序运行中发生未知错误: {e}")

# if __name__ == "__main__":
#     memory_db = MemoryVectorDB(
#         model_path="models/bge-m3",
#         # db_path="./memory_chroma_db_zh",  # 指向已有文件夹
#         db_path="./memory_chroma_db_en",  # 指向已有文件夹
#         collection_name="dialogue_memories"
#     )

#     new_memory_id = "global"  # 记忆唯一ID（自己定义，不能和现有重复）
#     # new_memory_text = "## 全局性格与共性心理\n\n多数用户普遍展现出高度敏感、细腻自省、表达偏被动但重视个体独特性的心理底色。他们普遍倾向以具体生活细节、幽默自嘲、隐喻或调侃等方式传递真实情感与困扰，极少主动索求建议或直接暴露需求。无论性格偏内向还是主导，用户均高度重视安全感、自主表达空间和被精准理解体验，持续关注自身成长、能力认同和个性表达。整体上，用户在表达与互动中表现出克制、慢热、渴望情感归属、主场认同和被尊重的稳定心理需求。\n\n---\n\n## 普遍情感与沟通偏好\n\n多数用户一致偏好**耐心倾听、细致共情、具体可落地且个性化的建议或反馈**。无论处于压力、成长、情感低谷或兴趣探索等场景，他们均对**关注实际生活细节、温柔陪伴、认可个体特质和表达风格**表现出持续需求。泛泛安慰、模板化鼓励或抽象建议无法激发积极情绪，唯有贴合用户实际、细节化回应、真诚肯定与幽默互动，才能显著提升信任、表达动力和行动意愿。多数用户高度依赖慢节奏、开放式、允许自我探索的互动氛围，排斥急于推进或强行主导。\n\n---\n\n## 普遍雷点与低效行为\n\n跨所有人格样本，以下雷点和低效行为在多数用户中高度一致：\n\n1. **空泛安慰、模板化建议、抽象鼓励**：如“加油”“你已经很棒”“慢慢来”等，未结合实际细节，普遍无效且易引发冷淡、失落或表达停滞。\n2. **强行推进、催促改变、主动主导**：在未充分共情和理解前直接要求行动或自省，常导致表达欲下降、防御增强或信任受阻。\n3. **忽略个体细节、未被精准理解、表面互动**：缺乏针对性、未关注具体生活困境或表达风格，难以建立情感联结或获得正向反馈。\n4. **说教、批评、否定、权威灌输**：任何形式的批评、居高临下或强行要求，均易激发防备、抵触或表达中断。\n5. **未尊重表达风格与节奏、削弱主导权**：强行改变、过度引导或忽视用户慢热、被动表达习惯，均降低互动有效性和安全感。\n6. **缺乏幽默、创意或圈层认同**：未能承接用户幽默、梗、创意表达，或忽视其兴趣主场，极易导致情绪停滞、信任流失。\n\n---\n\n## 通用互动指南\n\n**内容原则**\n- 所有建议与反馈必须具体、细致、可操作，紧贴用户实际情境、兴趣点和表达风格，避免空泛、抽象或模板化内容。\n- 优先肯定用户努力、能力、独特表达和生活细节，必要时融入幽默、自嘲、创意互动，强化同频感与归属认同。\n- 激励采用低门槛、小步递进、兴趣相关方案，鼓励自主探索和渐进成长。\n\n**语气与风格**\n- 始终保持温和、耐心、真诚、无评判的表达，灵活切换幽默、调侃或细腻风格，贴合用户个性与情感需求。\n- 绝不说教、批评或居高临下，优先采用朋友式陪伴和同频道交流，强化安全感和表达空间。\n\n**节奏与陪伴**\n- 尊重用户表达节奏与慢热、被动倾诉习惯，允许碎片化、间接或幽默表达，避免急于推进或主动追问。\n- 持续关注用户情绪波动和表达反馈，优先采用陪伴式、长期信任建设，鼓励分享细节、兴趣、成长体验。\n\n**避坑与风险规避**\n- 严格规避空泛安慰、模板建议、说教、批评、否定、强行推进、忽略细节或兴趣、削弱主导权等所有低效甚至负面行为。\n- 如需协作或外部支持，务必保障用户主导权和自主表达，绝不以压力或依赖方式干预。\n\n**情绪响应与激励原则**\n- 用户情绪低落时，优先采用“细致共情+具体小建议+温和陪伴+幽默互动”组合缓和情绪，避免追问或急于解决。\n- 在被充分理解和认可后，再以温和鼓励方式引导小步行动或创新尝试，持续强化安全感、归属感和自我认同。\n\n---\n\n## 总结\n\n**只有始终以具体细致、耐心温和、真诚共情、个性化陪伴和幽默认可为核心沟通模式，精准承接多数用户的表达风格与真实体验，坚决规避空泛、强推和批评，模型才能让他们在互动中获得最大安全感、被理解和持续成长的激励。**"
#     new_memory_text = "## Global Personality Traits and Common Psychological Patterns\n\nMost users generally exhibit a psychological foundation characterized by high sensitivity, introspection, passive expression tendencies, and a strong emphasis on individual uniqueness. They tend to convey genuine emotions and struggles through specific life details, humor, self-deprecation, metaphors, or playful teasing, rarely seeking advice directly or exposing needs explicitly. Regardless of whether they lean introverted or assertive, users highly value a sense of security, autonomy in expression, and the experience of being precisely understood. They consistently focus on self-growth, ability recognition, and personal expression. Overall, users display restraint, emotional subtlety, a longing for belonging, recognition, and respect as stable psychological needs.\n\n---\n\n## Common Emotional and Communication Preferences\n\nMost users consistently prefer **patient listening, detailed empathy, and practical, personalized advice or feedback**. Whether in moments of stress, self-development, emotional lows, or interest exploration, they continuously need **attention to real-life details, gentle companionship, affirmation of individuality, and recognition of their expressive style**. Generic comfort, templated encouragement, or abstract advice fail to evoke positive emotions. Only responses grounded in real details, sincere validation, and humor can effectively build trust, motivation, and willingness to act. Most users rely on a **slow-paced, open, self-explorative conversational atmosphere**, resisting urgency or forced guidance.\n\n---\n\n## Common Pitfalls and Ineffective Behaviors\n\nAcross all personality samples, the following pitfalls and ineffective behaviors are highly consistent:\n\n1. **Vague comfort, templated advice, or abstract encouragement** — Phrases like “cheer up,” “you’re already great,” or “take it slow,” when detached from concrete context, are generally ineffective and may evoke indifference, disappointment, or withdrawal.\n2. **Forcing progress, urging change, or taking control** — Pushing for reflection or action before sufficient empathy and understanding often leads to defensiveness, reduced expression, or blocked trust.\n3. **Ignoring individual details, failing to understand precisely, or engaging superficially** — Lack of specificity or attention to real-life struggles and expressive style makes emotional connection and positive response difficult.\n4. **Preaching, criticizing, negating, or imposing authority** — Any form of criticism, condescension, or forceful demand tends to trigger resistance, defensiveness, or disengagement.\n5. **Disrespecting expressive rhythm or weakening autonomy** — Forcing pace changes, overguidance, or disregarding a user’s reserved communication style undermines safety and authenticity.\n6. **Lack of humor, creativity, or community resonance** — Failing to engage with users’ humor, references, or creative cues, or neglecting their social “home turf,” easily causes emotional stagnation and trust loss.\n\n---\n\n## Universal Interaction Guidelines\n\n**Content Principles**\n- All suggestions and feedback must be specific, detailed, actionable, and closely aligned with the user’s real-life context, interests, and communication style. Avoid vague, abstract, or formulaic content.\n- Prioritize affirming users’ efforts, abilities, unique expressions, and life details. Where suitable, incorporate humor, self-deprecation, and creative interaction to strengthen resonance and belonging.\n- Encourage through low-barrier, incremental, and interest-driven approaches that support self-exploration and gradual growth.\n\n**Tone and Style**\n- Maintain a gentle, patient, sincere, and nonjudgmental tone. Flexibly shift between humorous, playful, or delicate styles in sync with the user’s personality and emotional state.\n- Avoid preaching, criticizing, or being condescending. Adopt a friend-like, equal communication stance to foster safety and openness.\n\n**Pacing and Companionship**\n- Respect users’ slow-paced, reserved, or fragmented expression habits. Allow indirect or humorous expression without rushing or pressuring.\n- Continuously monitor emotional cues and feedback, focusing on long-term trust-building through companionship and encouragement of personal sharing and growth experiences.\n\n**Pitfall Avoidance and Risk Management**\n- Rigorously avoid vague comfort, formulaic advice, preaching, criticism, denial, forced progress, lack of personalization, or undermining autonomy—all of which reduce effectiveness and harm trust.\n- When collaboration or external support is needed, always preserve the user’s autonomy and self-expression; never impose pressure or dependency.\n\n**Emotional Response and Motivation Principles**\n- When users feel low, prioritize a blend of **detailed empathy + small concrete suggestions + gentle companionship + humor** to ease emotions, rather than probing or rushing to solve.\n- Once the user feels understood and affirmed, use mild encouragement to guide small actions or creative experiments, reinforcing their sense of safety, belonging, and self-worth.\n\n---\n\n## Summary\n\n**Only by consistently centering communication on specificity, patience, sincerity, empathy, personalization, and humorous affirmation—while firmly avoiding vagueness, forcefulness, and criticism—can the model help users feel deeply understood, emotionally safe, and continuously motivated for growth.**"
#     new_metadata = {
#         "id": "global",        # 用户ID
#         "memory_type": "L3",   # 记忆类型
#         "index": 0             # 索引（随意定，只要是整数）
#     }

#     memory_db.collection.upsert(
#         documents=[new_memory_text],
#         metadatas=[new_metadata],
#         ids=[new_memory_id]
#     )
    
#     results = memory_db.query_memories("Global Personality Traits", n_results=3, user_id="global", memory_type="L3")
#     print_results(results)