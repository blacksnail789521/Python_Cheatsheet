import networkx as nx
import matplotlib.pyplot as plt
import json

def plot_G(G, pos, title = '', rad = 0.2, node_color = '#1f78b4',
           show_node_name_list = [], show_edge_name_list = []):
    
    nx.draw(G, pos, with_labels = (not show_node_name_list), 
            node_size = 1000, font_size = 20, #node_color = node_color,
            connectionstyle = f'arc3, rad = {rad}', arrowsize = 30)
    if show_node_name_list:
        labels = { node_id: { k:v for k, v in attr.items() if k in show_node_name_list } \
                   for node_id, attr in G.nodes(data=True) }
        nx.draw_networkx_labels(G, pos, labels)
    if show_edge_name_list:
        edge_labels = { (source, dest): { k:v for k, v in attr.items() if k in show_edge_name_list } \
                        for source, dest, attr in G.edges(data=True) }
        nx.draw_networkx_edge_labels(G, pos, edge_labels, label_pos = 0.3)
    plt.title(title, size = 16)
    plt.show()


if __name__ == '__main__':
    
    '''---------------------------------------------------------------------'''
    show_node_name_list = []
    # show_node_name_list = ['namespace']
    # show_node_name_list = ['namespace', 'workload']
    
    # show_edge_name_list = []
    # show_edge_name_list = ['id']
    show_edge_name_list = ['id', 'index']
    '''---------------------------------------------------------------------'''
    
    G = nx.DiGraph()
    G.add_nodes_from([
        ('1', {'namespace': ['1n'], 'workload': {1: '1w'}}),
        ('2', {'namespace': ['2n'], 'workload': '2w'}),
        ('3', {'namespace': ['3n'], 'workload': '3w'}),
    ])
    G.add_edges_from([
        ('1', '2', {'id': '1->2', 'index': 1}),
        ('2', '1', {'id': '2->1', 'index': 2}),
        ('2', '3', {'id': '2->3', 'index': 3}),
    ])

    pos = nx.circular_layout(G)    
    plot_G(G, pos, 'G', show_node_name_list = show_node_name_list, 
           show_edge_name_list = show_edge_name_list)