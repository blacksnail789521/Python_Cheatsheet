import networkx as nx


def remove_node_attr(G, unwanted_node_attr_list):
    
    for node_id, node_attr in G.nodes(data=True):
        [ node_attr.pop(unwanted_node_attr, None) for unwanted_node_attr in unwanted_node_attr_list ]
    
    # Check if we still have some unwanted node attributes
    for node_id, node_attr in G.nodes(data=True):
        assert len( set(unwanted_node_attr_list) & set(node_attr.keys()) ) == 0, \
               'We still have some unwanted node attributes!'
    
    return G


def remove_edge_attr(G, unwanted_edge_attr_list):
    
    for source, dest, edge_attr in G.edges(data=True):
        [ edge_attr.pop(unwanted_edge_attr, None) for unwanted_edge_attr in unwanted_edge_attr_list ]
    
    # Check if we still have some unwanted edge attributes
    for source, dest, edge_attr in G.edges(data=True):
        assert len( set(unwanted_edge_attr_list) & set(edge_attr.keys()) ) == 0, \
               'We still have some unwanted edge attributes!'
    
    return G
    
if __name__ == '__main__':
    
    G = nx.DiGraph()
    G.add_nodes_from([
        ('1', {'namespace': ['1n'], 'workload': {1: '1w'}}),
        ('2', {'namespace': ['2n'], 'workload': '2w'}),
        ('3', {'namespace': ['3n'], 'workload': '3w'}),
    ])
    G.add_edges_from([
        ('1', '2', {'id': '1->2', 'index': 1, 'df': None}),
        ('2', '1', {'id': '2->1', 'index': 2}),
        ('2', '3', {'id': '2->3', 'index': 3}),
    ])
    
    G = remove_node_attr(G, ['namespace', 'not_exist'])
    G = remove_edge_attr(G, ['index', 'df', 'not_exist'])
    
    print(list(G.nodes(data=True))[0]) # ('1', {'workload': {1: '1w'}})
    print(list(G.edges(data=True))[0]) # ('1', '2', {'id': '1->2'})