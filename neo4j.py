# neo4j_manager.py
from py2neo import Graph
from itertools import combinations
from config import set_argument

class Neo4jManager:
    def __init__(self):
        args = set_argument()
        self.graph = Graph(args.neo4j_url, auth=(args.neo4j_usr, args.neo4j_pwd))

    def get_kg_schema(self):
        """获取图谱结构信息"""
        schema_info = {'node_labels': [], 'relationship_types': []}
        try:
            labels = self.graph.run("CALL db.labels()").data()
            schema_info['node_labels'] = [l['label'] for l in labels]
            rels = self.graph.run("CALL db.relationshipTypes()").data()
            schema_info['relationship_types'] = [r['relationshipType'] for r in rels]
            return schema_info
        except Exception as e:
            print(f"❌ Schema Error: {e}")
            return None

    def query_one_hop_relations_by_id(self, node_id, limit=3):
        """
        [单跳查询] 基于 ID 查邻居
        返回格式: "StartName -[Type]-> NeighborName"
        """
        cypher = """
        MATCH (start)-[r]-(neighbor)
        WHERE id(start) = $node_id
        RETURN 
            type(r) as relation_type,
            start.name as start_name,
            neighbor.name as neighbor_name
        LIMIT $limit
        """
        try:
            results = self.graph.run(cypher, node_id=node_id, limit=limit).data()
            relations = []
            seen = set()
            for record in results:
                # 构造易读的字符串证据
                sig = f"{record['start_name']} -[{record['relation_type']}]- {record['neighbor_name']}"
                if sig not in seen:
                    relations.append(sig)
                    seen.add(sig)
            return relations
        except Exception as e:
            print(f"❌ One-Hop Error (ID: {node_id}): {e}")
            return []

    def query_shortest_path_by_ids(self, start_id, end_id):
        """
        [最短路径] 基于 ID 查路径
        返回包含节点名称和关系的路径结构
        """
        # 使用 shortestPath 算法，最大跳数设为 4 (过深的关系对 RAG 意义不大且慢)
        cypher = """
        MATCH (s), (e)
        WHERE id(s) = $start_id AND id(e) = $end_id
        MATCH p = shortestPath((s)-[*..4]-(e))
        RETURN p
        LIMIT 1
        """
        try:
            results = self.graph.run(cypher, start_id=start_id, end_id=end_id).data()
            if not results:
                return None
            
            path = results[0]['p']
            nodes_info = []
            rels_info = []

            # 提取易读的节点信息
            for node in path.nodes:
                nodes_info.append({
                    'id': node.identity,
                    'name': node.get('name', 'Unknown'),
                    'labels': list(node.labels)
                })
            
            # 提取易读的关系信息
            for rel in path.relationships:
                rels_info.append({
                    'type': type(rel).__name__,
                    'start_name': rel.start_node.get('name'),
                    'end_name': rel.end_node.get('name')
                })
            return {
                'path_str': f"Path found between {nodes_info[0]['name']} and {nodes_info[-1]['name']}",
                'nodes': nodes_info,
                'relationships': rels_info,
                'length': len(rels_info)
            }
        except Exception as e:
            # print(f"Path Error: {e}")
            return None

    def query_subgraph_by_ids(self, node_ids):
        """
        [核心入口] 输入 ID 列表，自动计算两两最短路 + 单跳信息
        """
        res = {}
        
        # 1. 单个节点的 One-Hop 信息
        for nid in node_ids:
            # 先查一下这个 ID 对应的名字，方便做 Key
            name_cypher = "MATCH (n) WHERE id(n)=$nid RETURN n.name as name"
            try:
                name_res = self.graph.run(name_cypher, nid=nid).data()
                if not name_res: continue
                node_name = name_res[0]['name']
                
                # 查单跳
                one_hop = self.query_one_hop_relations_by_id(nid)
                if one_hop:
                    res[f"{node_name} (One-Hop)"] = one_hop
            except:
                continue

        # 2. 两两节点的最短路径 (Combinations)
        if len(node_ids) > 1:
            target_ids = node_ids[:6] 
            for start_id, end_id in combinations(target_ids, 2):
                path_info = self.query_shortest_path_by_ids(start_id, end_id)
                if path_info:
                    # 获取名字用于 Key
                    s_name = path_info['nodes'][0]['name']
                    e_name = path_info['nodes'][-1]['name']
                    res[f"Path: {s_name} <-> {e_name}"] = path_info

        return res