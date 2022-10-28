import networkx as nx

from plot_graph import plot_G


G = nx.DiGraph()
G.add_nodes_from([
    ('1', {'namespace': 1, 'workload': 1}),
    ('2', {'namespace': 2, 'workload': 2}),
    ('3', {'namespace': 3, 'workload': 3}),
])
G.add_edges_from([
    ('1', '2', {'id': '1->2'}),
    ('2', '3', {'id': '2->3'}),
])

pos = nx.circular_layout(G)
plot_G(G, pos, 'G')