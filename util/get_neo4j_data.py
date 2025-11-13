""" 根据获取到的实体信息，查询知识图谱中的节点信息，并获取最短路径和单跳关系 """
from py2neo import Graph
from itertools import combinations
graph = Graph("bolt://172.18.51.200:7687",auth=("neo4j","bitlab512"))
#graph = Graph("bolt://localhost:7687",auth=("neo4j","12345678"))
def query_nodes_info(data):
    entities = data['entities']
    nodes = []
    results = {}
    for entity in entities:
        entity_name = entity['standard_name']
        node = query_node_by_name(entity_name)
        if node:
            nodes.append(node)
        else:
            print(f"未找到实体{entity_name}")

    for start_node, end_node in combinations(nodes, 2):
        start_name = start_node['properties'].get('name')
        end_name = end_node['properties'].get('name')
        # print(f"查询{start_name}与{end_name}的最短路径")
        path_info = query_shortest_path(start_name, end_name)
        if path_info:
            results[f"{start_name}->{end_name}"] = path_info
        else:
            print(f"未找到{start_name}与{end_name}的最短路径")
            one_node_info = query_one_hop_relations(start_node,limit=50)
            results[f"{start_name}->{end_name}"] = one_node_info
    return results

def query_node_by_name(node_name):
    """ 根据名称查询节点 """
    cypher = f"MATCH (n) WHERE n.name = $name RETURN n,labels(n) as labels,properties(n) as properties"
    try:
        results = graph.run(cypher, name=node_name)
        data = results.data()
        if data:
            return{
                'id': data[0]['n'].identity,
                'labels': data[0]['labels'],
                'properties': data[0]['properties']
            }
    except Exception as e:
        print(f"查询节点{node_name}失败：{e}")
    return None

def query_shortest_path(start_name, end_name):
    """ 查询最短路径 """
    cypher = "MATCH p = shortestPath((start {name: $start_name})-[*..10]-(end {name: $end_name})) RETURN p"
    try:
        results = graph.run(cypher,start_name=start_name,end_name=end_name).data()
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

def query_one_hop_relations(node, limit=100):
    """ 查询单跳关系 """
    node_name = node['properties'].get('name')
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

        # print(f'找到{len(relationships)}条关系')
        return relationships
    except Exception as e:
        print(f"查询单跳关系失败：{e}")
    return []

