import networkx as nx

from plot_graph import plot_G


G = nx.DiGraph()
G.add_nodes_from([
    ("1", {"trigger_point": False}),
    ("2", {"trigger_point": True}),
    ("3", {"trigger_point": False}),
    ("4", {"trigger_point": False}),
    ("5", {"trigger_point": False}),
])
G.add_edges_from([
    ("1", "2"),
    ("2", "3"),
    ("4", "5"),
])

pos = nx.circular_layout(G)
plot_G(G, pos, 'G')

print(nx.has_path(G, "1", "3")) # True
print(nx.has_path(G, "2", "1")) # False
print(nx.has_path(G, "1", "4")) # False