# 读取知识图谱数据并构建本地向量数据库索引
import pickle, faiss
import numpy as np
from py2neo import Graph
from itertools import combinations
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from config import set_argument
args = set_argument()
graph = Graph(args.neo4j_url, auth=(args.neo4j_usr, args.neo4j_pwd))

class Neo4jManager:
    def __init__(self):
        self.graph = graph

    def get_kg_schema(self):
        # 获取知识图谱的结构信息
        schema_info = {
            'node_labels': [],
            'relationship_types': [],
            'property_keys': []
        }
        try:
            labels_cypher = "CALL db.labels()"
            label_results = self.graph.run(labels_cypher).data()
            schema_info['node_labels'] = [label['label'] for label in label_results]

            rels_cypher = "CALL db.relationshipTypes()"
            rel_results = self.graph.run(rels_cypher).data()
            schema_info['relationship_types'] = [rel['relationshipType'] for rel in rel_results]

            return schema_info
        except Exception as e:
            print(e)
            return None
        
    def query_shortest_path(self, start_name, end_name):
        # 最短路查询（已针对同名节点去重优化）
        
        # 逻辑解释：
        # 1. MATCH (s), (e): 先找出所有叫 start_name 的节点集合 s，和所有叫 end_name 的节点集合 e。
        #    如果数据库里 "Transitional cell carcinoma..." 有2个节点，s 就会包含这两个节点。
        # 2. MATCH p = shortestPath((s)-[*..10]-(e)): 计算所有 s 和 e 组合之间的最短路径。
        # 3. ORDER BY length(p): 按路径长度排序。
        # 4. LIMIT 1: 只取最短的那一条。
        
        cypher = """
        MATCH (s {name: $start_name}), (e {name: $end_name})
        WHERE id(s) <> id(e)
        MATCH p = shortestPath((s)-[*..10]-(e))
        RETURN p
        ORDER BY length(p) ASC
        LIMIT 1
        """
        
        try:
            results = self.graph.run(cypher, start_name=start_name, end_name=end_name).data()
            
            if not results:
                # print(f"未找到 {start_name} 和 {end_name} 之间的路径")
                return None
            
            # 解析路径数据
            path = results[0]['p']
            nodes_info = []
            rels_info = []
            
            # 提取节点信息
            for node in path.nodes:
                nodes_info.append({
                    'id': node.identity,          # Neo4j 内部 ID
                    'labels': list(node.labels),  # 节点标签
                    'properties': dict(node)      # 节点属性
                })
            
            # 提取关系信息
            for rel in path.relationships:
                rels_info.append({
                    'id': rel.identity,
                    'start_node': rel.start_node.identity,
                    'end_node': rel.end_node.identity,
                    'type': type(rel).__name__,   # 增加关系类型名称
                    'properties': dict(rel)
                })
                
            return {
                'path_nodes': nodes_info,
                'path_rels': rels_info,
                'length': len(rels_info) # 方便后续查看跳数
            }
            
        except Exception as e:
            print(f"查询最短路径失败 ({start_name} -> {end_name}): {e}")
            return None
    
    def query_one_hop_relations(self,node_name, limit=5):
        # 查询单跳关系
        cypher = """
        MATCH (start {name: $node_name})-[r]-(neighbor)
        RETURN 
            type(r) as relation_type,
            start.name as start_name,
            neighbor.name as neighbor_name,
            properties(neighbor) as neighbor_props
        LIMIT $limit
        """
        
        try:
            results = self.graph.run(cypher, node_name=node_name, limit=limit).data()
            unique_relations = set()
            relation_strings = []
            
            for record in results:
                rel_type = record['relation_type']
                neighbor = record['neighbor_name']
                # 构建一个唯一的字符串签名
                rel_sig = f"{node_name} -[{rel_type}]- {neighbor}"
                
                if rel_sig not in unique_relations:
                    unique_relations.add(rel_sig)
                    relation_strings.append(rel_sig)
            
            return relation_strings
            
        except Exception as e:
            print(f"查询单跳关系失败：{e}")
            return []
    
    def query_node_info(self,entity_list):
        nodes = []
        res = {}
        for entity in entity_list:
            cypher = f"MATCH (n) WHERE n.name = '{entity}' RETURN n,labels(n) as labels, properties(n) as properties"
            result = self.graph.run(cypher).data()
            if result:
                nodes.append({'id': result[0]['n'].identity,'labels': result[0]['labels'],'properties': result[0]['properties']})
            # else:
            #     print(f"No node found for {entity}")
        if not nodes:
            return res
        if len(nodes) == 1:
            entity_name = nodes[0]['properties'].get('name')
            one_top_info = self.query_one_hop_relations(entity_name)
            res[f"{entity_name} one-hop info"] = one_top_info
            return res
        for start_node,end_node in combinations(nodes,2):
            start_name = start_node['properties'].get('name')
            end_name = end_node['properties'].get('name')
            if start_name and end_name:
                path_info = self.query_shortest_path(start_name, end_name)
                if path_info:
                    res[f"{start_name}-{end_name}"] = path_info
                else:
                    # print(f"No path found for {start_name} and {end_name}")
                    one_top_info = self.query_one_hop_relations(start_name)
                    res[f"{start_name} one-hop info"] = one_top_info
                    one_top_info = self.query_one_hop_relations(end_name)
                    res[f"{end_name} one-hop info"] = one_top_info
        return res
    
    def fetch_all_nodes_generator(self, batch_size=10000):
        """ 生成器分批读取所有节点 """
        offset = 0
        base_cypher = "MATCH (n) WHERE n.name IS NOT NULL"
        while True:
            cypher = f"""
            {base_cypher}
            RETURN id(n) as id, n.name as name, labels(n) as labels
            ORDER BY id(n)
            SKIP {offset}
            LIMIT {batch_size}
            """
            try:
                results = self.graph.run(cypher).data()
                if not results:
                    break
                yield results
                offset += batch_size
            except Exception as e:
                print(f"查询节点失败：{e}")
                break
    
    def build_local_vector_db():
        # 构建本地向量数据库索引
        MODEL_NAME = args.embedding_model_en
        INDEX_FILE = "output/faiss_index_neo4j.index"
        META_FILE = "output/entity_metadata.pkl"
        BATCH_SIZE = 1000
        
        manager = Neo4jManager()
        model = SentenceTransformer(MODEL_NAME)
        embedding_dim = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(embedding_dim)
        metadata_map = []

        node_generator = manager.fetch_all_nodes_generator(batch_size=BATCH_SIZE)
        total_processed = 0
        for batch_nodes in node_generator:
            batch_names = [node['name'] for node in batch_nodes]
            embeddings = model.encode(batch_names, show_progress_bar=True)

            faiss.normalize_L2(embeddings)
            index.add(np.array(embeddings).astype('float32'))

            for node in batch_nodes:
                metadata_map.append({
                    'neo4j_id': node['id'],
                    'name': node['name'],
                    'labels': node['labels']
                })
            total_processed += len(batch_nodes)
            print(f"Processed {total_processed} nodes.")
        faiss.write_index(index, INDEX_FILE)
        with open(META_FILE, 'wb') as f:
            pickle.dump(metadata_map, f)
        print(f"Local vector database built successfully.")

