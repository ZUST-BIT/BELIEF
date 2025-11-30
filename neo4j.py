from config import set_argument
from py2neo import Graph
from itertools import combinations
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
        # 最短路查询
        cypher = "MATCH p = shortestPath((start {name: $start_name})-[*..10]-(end {name: $end_name})) RETURN p"
        try:
            results = self.graph.run(cypher,start_name=start_name,end_name=end_name).data()
            if not results:
                return None
            path = results[0]['p']
            nodes_info = []
            rels_info = []
            for node in path.nodes:
                nodes_info.append({
                    'id': node.identity,
                    'labels': list(node.labels),
                    'properties': dict(node)
                })
            for rel in path.relationships:
                rels_info.append({
                    'id': rel.identity,
                    'start_node': rel.start_node.identity,
                    'end_node':rel.end_node.identity,
                    'properties':dict(rel)
                })
            return {
                'path_nodes':nodes_info,
                'path_rels':rels_info
            }
        except Exception as e:
            print(f"查询最短路径失败：{e}")
        return None
    
    def query_one_hop_relations(self,node_name, limit=5):
        # 查询单跳关系
        cypher_out = """
        MATCH (start)-[r]->(neighbor)
        WHERE start.name = $node_name
        RETURN
            'OUT' as direction,
            type(r) as relation_type,
            properties(r) as rel_props,
            labels(neighbor) as neighbor_labels,
            properties(neighbor) as neighbor_props,
            neighbor.name as neighbor_name

        LIMIT $limit
        """

        cypher_in = """
        MATCH (start)<-[r]-(neighbor)
        WHERE start.name = $node_name
        RETURN
            'IN' as direction,
            type(r) as relation_type,
            properties(r) as rel_props,
            labels(neighbor) as neighbor_labels,
            properties(neighbor) as neighbor_props,
            neighbor.name as neighbor_name

        LIMIT $limit
        """
        try:
            relationships = []
            results_out = graph.run(cypher_out, node_name=node_name, limit=limit)
            for record in results_out:
                rel_type = record['relation_type']
                neighbor_name = record['neighbor_name'] or 'Unknown'
                rel_string = f"{node_name} -[{rel_type}]-> {neighbor_name}"
                relationships.append(rel_string)
            
            results_in = graph.run(cypher_in, node_name=node_name, limit=limit)
            for record in results_in:
                rel_type = record['relation_type']
                neighbor_name = record['neighbor_name'] or 'Unknown'
                rel_string = f"{node_name} <-[{rel_type}]- {neighbor_name}"
                relationships.append(rel_string)
            return relationships
        except Exception as e:
            print(f"查询单跳关系失败：{e}")
        return []
    
    def query_node_info(self,entity_list):
        nodes = []
        res = {}
        for entity in entity_list:
            print(f"Querying node info for {entity}")
            cypher = f"MATCH (n) WHERE n.name = '{entity}' RETURN n,labels(n) as labels, properties(n) as properties"
            result = self.graph.run(cypher).data()
            if result:
                nodes.append({'id': result[0]['n'].identity,'labels': result[0]['labels'],'properties': result[0]['properties']})
            else:
                print(f"No node found for {entity}")
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
            print(f"Querying path between {start_name} and {end_name}")
            if start_name and end_name:
                print(f"Querying path between {start_name} and {end_name}")
                path_info = self.query_shortest_path(start_name, end_name)
                if path_info:
                    res[f"{start_name}-{end_name}"] = path_info
                else:
                    print(f"No path found for {start_name} and {end_name}")
                    one_top_info = self.query_one_hop_relations(start_name)
                    res[f"{start_name} one-hop info"] = one_top_info
                    one_top_info = self.query_one_hop_relations(end_name)
                    res[f"{end_name} one-hop info"] = one_top_info
        return res
    