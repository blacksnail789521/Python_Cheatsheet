import networkx as nx


# Create a graph
G = nx.DiGraph()
G.add_nodes_from([
    (1, {'namespace': 1, 'workload': 1}),
    (2, {'namespace': 2, 'workload': 2}),
    (3, {'namespace': 3, 'workload': 3}),
])
G.add_edges_from([
    (1, 2, {'id': '1->2'}),
    (2, 3, {'id': '2->3'}),
])

# Update one node attribute
for node_id, node_attr in G.nodes(data = True):
    if node_attr['namespace'] == 2:
        #node_attr['new_node_attr'] = 'XDXD' # Feasible but not a very good practice
        G.nodes[node_id]['new_node_attr'] = 'XDXD'
        break
print(G.nodes[2])
'''
{'namespace': 2, 'workload': 2, 'new_node_attr': 'XDXD'}
'''

# Update one edge attribute
for source, dest, edge_attr in G.edges(data = True):
    if edge_attr['id'] == '2->3':
        # edge_attr['new_edge_attr'] = 'xdxd' # Feasible but not a very good practice
        G.edges[source, dest]['new_edge_attr'] = 'xdxd'
        break
print(G.edges[2, 3])
'''
{'id': '2->3', 'new_edge_attr': 'xdxd'}
'''