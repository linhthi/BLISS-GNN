
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void sample_neighbors_v2(self, int node, vector[int]& edges) nogil:
    cdef int degree
    cdef int edge_size
    cdef vector[int]* neighbors = &self.adj[node]
    degree = self.degree[node]
    edge_size = edges.size()
    cdef vector[int].iterator it
    cdef int i
    cdef vector[int] samples
    cdef vector[double] probs
    cdef vector[double]* sample_probs = &self.sample_probs[node]
    if degree <= self.neighbor_limit:
        edges.resize(edge_size + degree*2)
        i = 0
        while i < degree:
            edges[edge_size+2*i] = node
            edges[edge_size+2*i+1] = deref(neighbors)[i]
            inc(it)
            i += 1
    else:
        edges.resize(edge_size + self.neighbor_limit*2)
        samples = random_choice(deref(neighbors), deref(sample_probs), self.neighbor_limit)

        i = 0
        while i < self.neighbor_limit:
            edges[edge_size+2*i] = node
            edges[edge_size+2*i+1] = samples[i]
            i += 1