

import easygraph as eg



G = eg.Graph()
G.add_edges([(1,2),(7,8),(8,10)])
print(list(G.neighbors(7)))
for n in G.edges:
    print(n)