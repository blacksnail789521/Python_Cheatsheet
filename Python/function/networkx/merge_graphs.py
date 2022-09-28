import networkx as nx

from plot_graph import plot_G


G1 = nx.DiGraph()
G1.add_nodes_from([
    ('1', {'namespace': 1, 'workload': 1}),
    ('2', {'namespace': 2, 'workload': 2}),
    ('3', {'namespace': 3, 'workload': 3}),
])
G1.add_edges_from([
    ('1', '2', {'id': '1->2'}),
    ('2', '3', {'id': '2->3'}),
])

pos = nx.circular_layout(G1)
plot_G(G1, pos, 'G1')
'''---------------------------------------------------------------------'''

G2 = nx.DiGraph()
G2.add_nodes_from([
    ('1', {'namespace': 1, 'workload': 1}),
    ('2', {'namespace': 2, 'workload': 2}),
    ('3', {'namespace': 3, 'workload': 3}),
])
G2.add_edges_from([
    ('2', '1', {'id': '2->1'}),
])

pos = nx.circular_layout(G2)
plot_G(G2, pos, 'G2')
'''---------------------------------------------------------------------'''

G3 = nx.DiGraph()
G3.add_nodes_from([
    ('1', {'namespace': 1, 'workload': 1}),
    ('2', {'namespace': 2, 'workload': 2}),
    ('3', {'namespace': 3, 'workload': 3}),
])
G3.add_edges_from([
    ('1', '2', {'id': '1->2'}),
    ('2', '1', {'id': '2->1'}),
    ('3', '2', {'id': '3->2'}),
])

pos = nx.circular_layout(G3)
plot_G(G3, pos, 'G3')
'''---------------------------------------------------------------------'''

aggregated_G = nx.compose_all([G1, G2, G3])
plot_G(aggregated_G, pos, 'aggregated_G')