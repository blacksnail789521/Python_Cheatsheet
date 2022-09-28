import networkx as nx

from plot_graph import plot_G


G = nx.DiGraph()
G.add_nodes_from([
    (1, {'pos': (0, 1)}), 
    (2, {'pos': (0, 0)}), 
    (3, {'pos': (1, 1)}), 
    (4, {'pos': (1, 0)}), 
    (5, {'pos': (2, 1)}), 
    (6, {'pos': (2, 0)}), 
    (7, {'pos': (3, 1)}), 
    (8, {'pos': (3, 0)}), 
    (9, {'pos': (4, 1)}), 
    (10, {'pos': (4, 0)}), 
])
G.add_edges_from([
    (1, 3),
    (1, 4),
    (2, 4),
    (3, 5),
    (4, 5),
    (4, 6),
    (5, 7),
    (5, 8),
    (7, 9),
    (7, 10),
])

pos = nx.get_node_attributes(G,'pos')
plot_G(G, pos, rad = 0, title = 'G')


# Upstream graph (including source node itself)
source_node = 5
upstream_G = G.subgraph([ n for n in nx.bfs_tree(G, source_node, reverse = True) ])
plot_G(upstream_G, pos, rad = 0, title = 'upstream_G')

# Downstream graph (including source node itself)
source_node = 5
downstream_G = G.subgraph([ n for n in nx.bfs_tree(G, source_node) ])
plot_G(downstream_G, pos, rad = 0, title = 'downstream_G')