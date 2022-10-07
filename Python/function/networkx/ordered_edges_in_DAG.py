import networkx as nx
import random
from pprint import pprint

from plot_graph import plot_G


# Create a graph with random ordered nodes and edges
node_list = [
    (1, {'pos': (0, 1)}), 
    (2, {'pos': (1, 1)}), 
    (3, {'pos': (1, 0)}), 
    (4, {'pos': (2, 1)}), 
    (5, {'pos': (2, 0)}), 
]
edge_list = [
    (1, 4, {'id': 1}),
    (1, 2, {'id': 2}),
    (1, 3, {'id': 3}),
    (2, 3, {'id': 4}),
    (2, 4, {'id': 5}),
    (3, 4, {'id': 6}),
    (3, 5, {'id': 7}),
]
random.shuffle(node_list)
random.shuffle(edge_list)
G = nx.DiGraph()
G.add_nodes_from(node_list)
G.add_edges_from(edge_list)

# Plot it
pos = nx.get_node_attributes(G,'pos')
plot_G(G, pos, 'G', show_edge_name_list = ['id'])

# Get ordered_edge_list
ordered_edge_list = [ edge for node in nx.topological_sort(G) for edge in G.edges(node, data = True) ]
pprint(ordered_edge_list)