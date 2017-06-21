import pygraph
from graphviz import Source
import random
import numpy as np


position =1

g= pygraph.UndirectedGraph()

q_tab = []


def graph_to_dot(graph, node_renderer=None, edge_renderer=None):
    """Produces a DOT specification string from the provided graph."""
    node_pairs = graph.nodes.items()
    edge_pairs = graph.edges.items()

    if node_renderer is None:
        node_renderer_wrapper = lambda nid: ''
    else:
        node_renderer_wrapper = lambda nid: ' [%s]' % ','.join(
            map(lambda tpl: '%s=%s' % tpl, node_renderer(graph, nid).items()))

    # Start the graph
    graph_string = 'digraph G {\n'
    graph_string += 'overlap=scale;\n'

    # Print the nodes (placeholder)
    for node_id, node in node_pairs:
        graph_string += '%i%s;\n' % (node_id, node_renderer_wrapper(node_id))

    # Print the edges
    for edge_id, edge in edge_pairs:
        node_a = edge['vertices'][0]
        node_b = edge['vertices'][1]
        graph_string += '%i -> %i;\n' % (node_a, node_b)

    # Finish the graph
    graph_string += '}'

    return graph_string




def build_graph():
	for i in range(9):
		g.new_node()

	g.nodes[1]['data']['rw'] = 5 # 0,0
	g.nodes[2]['data']['rw'] = 3 # 0,1
	g.nodes[3]['data']['rw'] = 9 # 0,2
	g.nodes[4]['data']['rw'] = 2 # 1,0
	g.nodes[5]['data']['rw'] = 5 # 1,1
	g.nodes[6]['data']['rw'] = 8 # 1,2
	g.nodes[7]['data']['rw'] = 3 # 2,0
	g.nodes[8]['data']['rw'] = 2 # 2,1
	g.nodes[9]['data']['rw'] = 8# 2,2

	g.new_edge(1, 2, cost=.3)
	g.new_edge(1, 4, cost=.2)

	g.new_edge(2, 5, cost=.1)
	g.new_edge(2, 3, cost=.6)

	g.new_edge(3, 6, cost=.7)

	g.new_edge(4, 5, cost=.3)
	g.new_edge(4, 7, cost=.1)

	g.new_edge(5, 6, cost=.9)
	g.new_edge(5, 8, cost=.8)

	g.new_edge(6, 9, cost=.5)

	g.new_edge(7, 8, cost=.3)
	g.new_edge(8, 9, cost=.1)

	g.new_edge(1, 6, cost=.1)



def print_graph():
	string=graph_to_dot(g)
	src = Source(string)
	src.render('C:/Users/wmalone/Desktop/Python/graph_struct/graph.gv', view=True) 


def move():
	global position
	global q_tab
	index = find_ele(q_tab,position)
	curr_neigh=g.neighbors(position)
	if index != -1:
		if np.random.uniform() < .9:
			next_move = check_q(index)
		else:
			next_move=random.choice(curr_neigh)
	else:
		next_move=random.choice(curr_neigh)
	

	if index != -1:
		if (q_tab[index][1][next_move-1] == 0):
			q_tab[index][1][next_move-1] = g.nodes[next_move]['data']['rw']
	else:
		row = position,[0,0,0,0,0,0,0,0,0]
		row[1][next_move-1]=g.nodes[next_move]['data']['rw']
		q_tab.append(row)

	
	
	#print('Position:',position,'Neighbors:',curr_neigh,'Next Move:',next_move)
	position=next_move

def check_q(index):
	mx = -1
	opt_move = -1
	curr_neigh=g.neighbors(position)
	q_line = q_tab[index][1]
	
	for i in curr_neigh:
		if q_line[i-1] > mx:
			mx = q_line[i-1]
			opt_move =i

	return opt_move

def print_q():
	print((0,[1,2,3,4,5,6,7,8,9]))
	for i in q_tab:
		print (i)

def find_ele(arr,ele):
	index = -1
	for i in arr:
		index+=1
		if i[0] == ele:
			return index
	return -1	

#main
build_graph()

i=0
j=0
inc =0
min_steps =9999999999999
for j in range(1000):
	while True:
		move()
		inc+=1
		if position == 9:
			if inc < min_steps:
				min_steps = inc
			break
	position = 1	
	inc=0	
print ('\nMin Steps: ',min_steps)

print_q()
print_graph()






