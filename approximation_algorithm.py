# Approximation algorithm
'''
The approximation algorithm is based on the algorithm described in the paper 
"Learning to order things" by W. W. Cohen, R. E. Schapire, Y. Singer (https://arxiv.org/abs/1105.5464)

The input is a preference graph: Gp = (V, E) s.t. each edge '(v,u)' has a label (epsilon, +, or -) and a weight
which is the predicted probability that the relation between vertices 'v' and 'u' is in a wide scope (+), 
narrow scope (-), or they are incoparable (epsilon).

The output is transitive directed acyclic graph (TDAG) which is a sub-graph of Gp that maximizes the sum of weights.
'''
import numpy as np

def get_vertices(Gp):
    '''
    Returns the list 'V' of vertices
    from a preference graph 'Gp'
    '''
    V = []
    for key in Gp.keys():
        V += key[0]
        V += key[1]
    return list(set(V))
        
def delta(Gp, u, v, label):
    '''
    Returns the probability of the edge (u,v)
    according to a label (epsilon, +, -)
    '''
    if label == '+':
        if (u,v) in Gp: return Gp[u,v][1]
        else: return Gp[v,u][2] 
    elif label == '-':
        if (u,v) in Gp: return Gp[(u,v)][2]
        else: return Gp[(v,u)][1] 
    else:
        if (u,v) in Gp: return Gp[(u,v)][0]
        else: return Gp[(v,u)][0] 

def pi(Gp, V, u):
    '''
    Returns the difference between the sum of
    outgoing and incoming edges of vertice 'u'
    '''
    S_out = sum([delta(Gp,u,v,'+') for v in V if u!=v])
    S_in = sum([delta(Gp,u,v,'-') for v in V if u!=v])
    return S_out - S_in 

# MAIN ALGORITHM
def create_pred_tdag(Gp):
    '''
    Returns a transitive directed acyclic graph 'G'
    from a given preference graph 'Gamma_p' (Gp)
    '''
    V = get_vertices(Gp)
    G = [] # TDAG

    Ranks = [] # list of tuples: (vertice, rank)
    rank = 0 # initial rank
    E = [] # set of edges

    while V:
        pi_u = [pi(Gp, V, u) for u in V]
        u_star = V[np.argmax(pi_u)]

        # assign_rank
        R = [v for (v,r) in Ranks if r == rank] # set of vertices with current rank
        same_rank = [delta(Gp, v, u_star, 'epsilon') > delta(Gp, v, u_star, '+') for v in R]
        if R == [] or all(same_rank):
            Ranks.append((u_star, rank))            
        else:
            rank += 1
            Ranks.append((u_star, rank))

        V.remove(u_star)

        # add_edges    
        R_star = [v for (v,r) in Ranks if r < rank] # set of vertices from all ranks before
        E += [(v, u_star) for v in R_star if delta(Gp, v, u_star, '+') > delta(Gp, v, u_star, 'epsilon')]

        G.append(u_star)
            
    return G, E
