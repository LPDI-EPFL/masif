# This code is derived work from Jake Vanderplas's Dijkstra code for SKLearn
# I modified this for MaSIF to extract patches, and also to compute polar coordinates. 
# Author: Jake Vanderplas  -- <vanderplas@astro.washington.edu>
# Modified and Derived by : Pablo Gainza for MaSIF - including angles.
# License: BSD 3 clause, (C) 2011

# cython: unraisable_tracebacks=True
import numpy as np
cimport numpy as np
import sys

from scipy.sparse import csr_matrix, isspmatrix, isspmatrix_csr

cimport cython
from libc.stdlib cimport malloc, free

np.import_array()

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

ITYPE = np.int32
ctypedef np.int32_t ITYPE_t
from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI, atan2, fmod

@cython.boundscheck(False)
cdef DTYPE_t average_angles(DTYPE_t angle1, DTYPE_t angle2, DTYPE_t weight1, DTYPE_t weight2):
    """Average (mean) of angles

    Return the average of an input sequence of angles. The result is between
    ``0`` and ``2 * math.pi``.
    If the average is not defined (e.g. ``average_angles([0, math.pi]))``,
    a ``ValueError`` is raised.
    """
    cdef DTYPE_t x, y
    cdef DTYPE_t mean_angle

    x = weight1*cos(angle1) + weight2*cos(angle2) 
    y = weight1*sin(angle1) + weight2*sin(angle2)

    # To get outputs from -pi to +pi, delete everything but math.atan2() here.
    mean_angle = fmod(atan2(y,x)+ 2*M_PI, 2*M_PI)
    
    return mean_angle

@cython.boundscheck(False)
cdef np.ndarray dijkstra(dist_matrix, 
        np.ndarray[ITYPE_t, ndim=2] patch_indices,
        np.ndarray[DTYPE_t, ndim=2] patch_rho,
        np.ndarray[DTYPE_t, ndim=2] patch_theta,
        theta0, 
        theta0_v_idx):
    """
    Dijkstra algorithm using Fibonacci Heaps
    Parameters
    ----------
    dist_matrix: sparse matrix
        dist_matrix is the csr_matrix of distances between connected points.
        unconnected points have distance=0.  
    patch_indices: ndarray
        on exit, patch_indices is overwritten with the outputted patch indices
    patch_rho: ndarray
        on exit, patch_rho is overwritten with the outputted patch indices' distance to the center of each patch
    Returns
    -------
        patch_rho, patch_indices
    """
    cdef unsigned int N = patch_indices.shape[0]
    cdef unsigned int i

    cdef FibonacciHeap heap

    cdef FibonacciNode* nodes = <FibonacciNode*> malloc(N *
                                                        sizeof(FibonacciNode))

    cdef np.ndarray distances, neighbors, indptr
    cdef np.ndarray distances2, neighbors2, indptr2

    if not isspmatrix_csr(dist_matrix):
        dist_matrix = csr_matrix(dist_matrix)

    distances = np.asarray(dist_matrix.data, dtype=DTYPE, order='C')
    neighbors = np.asarray(dist_matrix.indices, dtype=ITYPE, order='C')
    indptr = np.asarray(dist_matrix.indptr, dtype=ITYPE, order='C')

    for i from 0 <= i < N:
        initialize_node(&nodes[i], i)

    heap.min_node = NULL

    #use the csr -> csc sparse matrix conversion to quickly get
    # both directions of neighbors
    dist_matrix_T = dist_matrix.T.tocsr()

    distances2 = np.asarray(dist_matrix_T.data,dtype=DTYPE, order='C')
    neighbors2 = np.asarray(dist_matrix_T.indices,dtype=ITYPE, order='C')
    indptr2 = np.asarray(dist_matrix_T.indptr,dtype=ITYPE, order='C')

    for i from 0 <= i < N:
        dijkstra_one_row(i, neighbors, distances, indptr,
                neighbors2, distances2, indptr2,
                patch_indices, patch_rho, patch_theta, theta0[i], theta0_v_idx[i],
                &heap, nodes)

    free(nodes)

    return patch_indices


######################################################################
# FibonacciNode structure
#  This structure and the operations on it are the nodes of the
#  Fibonacci heap.
#

cdef struct FibonacciNode:
    unsigned int index
    unsigned int rank
    unsigned int state
    DTYPE_t val
    DTYPE_t theta_val
    FibonacciNode* parent
    FibonacciNode* left_sibling
    FibonacciNode* right_sibling
    FibonacciNode* children


cdef void initialize_node(FibonacciNode* node,
                          unsigned int index,
                          DTYPE_t val=0,
                          DTYPE_t theta_val=0):
    # Assumptions: - node is a valid pointer
    #              - node is not currently part of a heap
    node.index = index
    node.val = val
    node.theta_val= theta_val
    node.rank = 0
    node.state = 0  # 0 -> NOT_IN_HEAP

    node.parent = NULL
    node.left_sibling = NULL
    node.right_sibling = NULL
    node.children = NULL


cdef FibonacciNode* rightmost_sibling(FibonacciNode* node):
    # Assumptions: - node is a valid pointer
    cdef FibonacciNode* temp = node
    while(temp.right_sibling):
        temp = temp.right_sibling
    return temp


cdef FibonacciNode* leftmost_sibling(FibonacciNode* node):
    # Assumptions: - node is a valid pointer
    cdef FibonacciNode* temp = node
    while(temp.left_sibling):
        temp = temp.left_sibling
    return temp


cdef void add_child(FibonacciNode* node, FibonacciNode* new_child):
    # Assumptions: - node is a valid pointer
    #              - new_child is a valid pointer
    #              - new_child is not the sibling or child of another node
    new_child.parent = node

    if node.children:
        add_sibling(node.children, new_child)
    else:
        node.children = new_child
        new_child.right_sibling = NULL
        new_child.left_sibling = NULL
        node.rank = 1


cdef void add_sibling(FibonacciNode* node, FibonacciNode* new_sibling):
    # Assumptions: - node is a valid pointer
    #              - new_sibling is a valid pointer
    #              - new_sibling is not the child or sibling of another node
    cdef FibonacciNode* temp = rightmost_sibling(node)
    temp.right_sibling = new_sibling
    new_sibling.left_sibling = temp
    new_sibling.right_sibling = NULL
    new_sibling.parent = node.parent
    if new_sibling.parent:
        new_sibling.parent.rank += 1


cdef void remove(FibonacciNode* node):
    # Assumptions: - node is a valid pointer
    if node.parent:
        node.parent.rank -= 1
        if node.left_sibling:
            node.parent.children = node.left_sibling
        elif node.right_sibling:
            node.parent.children = node.right_sibling
        else:
            node.parent.children = NULL

    if node.left_sibling:
        node.left_sibling.right_sibling = node.right_sibling
    if node.right_sibling:
        node.right_sibling.left_sibling = node.left_sibling

    node.left_sibling = NULL
    node.right_sibling = NULL
    node.parent = NULL


######################################################################
# FibonacciHeap structure
#  This structure and operations on it use the FibonacciNode
#  routines to implement a Fibonacci heap

ctypedef FibonacciNode* pFibonacciNode


cdef struct FibonacciHeap:
    FibonacciNode* min_node
    pFibonacciNode[100] roots_by_rank  # maximum number of nodes is ~2^100.


cdef void insert_node(FibonacciHeap* heap,
                      FibonacciNode* node):
    # Assumptions: - heap is a valid pointer
    #              - node is a valid pointer
    #              - node is not the child or sibling of another node
    if heap.min_node:
        add_sibling(heap.min_node, node)
        if node.val < heap.min_node.val:
            heap.min_node = node
    else:
        heap.min_node = node


cdef void decrease_val(FibonacciHeap* heap,
                       FibonacciNode* node,
                       DTYPE_t newval):
    # Assumptions: - heap is a valid pointer
    #              - newval <= node.val
    #              - node is a valid pointer
    #              - node is not the child or sibling of another node
    node.val = newval
    if node.parent and (node.parent.val >= newval):
        remove(node)
        insert_node(heap, node)
    elif heap.min_node.val > node.val:
        heap.min_node = node


cdef void link(FibonacciHeap* heap, FibonacciNode* node):
    # Assumptions: - heap is a valid pointer
    #              - node is a valid pointer
    #              - node is already within heap

    cdef FibonacciNode *linknode
    cdef FibonacciNode *parent
    cdef FibonacciNode *child

    if heap.roots_by_rank[node.rank] == NULL:
        heap.roots_by_rank[node.rank] = node
    else:
        linknode = heap.roots_by_rank[node.rank]
        heap.roots_by_rank[node.rank] = NULL

        if node.val < linknode.val or node == heap.min_node:
            remove(linknode)
            add_child(node, linknode)
            link(heap, node)
        else:
            remove(node)
            add_child(linknode, node)
            link(heap, linknode)


cdef FibonacciNode* remove_min(FibonacciHeap* heap):
    # Assumptions: - heap is a valid pointer
    #              - heap.min_node is a valid pointer
    cdef FibonacciNode *temp
    cdef FibonacciNode *temp_right
    cdef FibonacciNode *out
    cdef unsigned int i

    # make all min_node children into root nodes
    if heap.min_node.children:
        temp = leftmost_sibling(heap.min_node.children)
        temp_right = NULL

        while temp:
            temp_right = temp.right_sibling
            remove(temp)
            add_sibling(heap.min_node, temp)
            temp = temp_right

        heap.min_node.children = NULL

    # choose a root node other than min_node
    temp = leftmost_sibling(heap.min_node)
    if temp == heap.min_node:
        if heap.min_node.right_sibling:
            temp = heap.min_node.right_sibling
        else:
            out = heap.min_node
            heap.min_node = NULL
            return out

    # remove min_node, and point heap to the new min
    out = heap.min_node
    remove(heap.min_node)
    heap.min_node = temp

    # re-link the heap
    for i from 0 <= i < 100:
        heap.roots_by_rank[i] = NULL

    while temp:
        if temp.val < heap.min_node.val:
            heap.min_node = temp
        temp_right = temp.right_sibling
        link(heap, temp)
        temp = temp_right

    return out





#@cython.boundscheck(False)
cdef void dijkstra_one_row(unsigned int i_node,
                    np.ndarray[ITYPE_t, ndim=1, mode='c'] neighbors1,
                    np.ndarray[DTYPE_t, ndim=1, mode='c'] distances1,
                    np.ndarray[ITYPE_t, ndim=1, mode='c'] indptr1,
                    np.ndarray[ITYPE_t, ndim=1, mode='c'] neighbors2,
                    np.ndarray[DTYPE_t, ndim=1, mode='c'] distances2,
                    np.ndarray[ITYPE_t, ndim=1, mode='c'] indptr2,
                    np.ndarray[ITYPE_t, ndim=2, mode='c'] patch_indices,
                    np.ndarray[DTYPE_t, ndim=2, mode='c'] patch_rho,
                    np.ndarray[DTYPE_t, ndim=2, mode='c'] patch_theta,
                    np.ndarray[DTYPE_t, ndim=1, mode='c'] theta0,
                    np.ndarray[ITYPE_t, ndim=1, mode='c'] theta0_v_idx,
                    FibonacciHeap* heap,
                    FibonacciNode* nodes):
    """
    Calculate distances from a single point to all targets using an
    undirected graph.
    Parameters
    ----------
    i_node : index of source point
    neighbors[1,2] : array, shape = [N,]
        indices of neighbors for each point
    distances[1,2] : array, shape = [N,]
        lengths of edges to each neighbor
    indptr[1,2] : array, shape = (N+1,)
        the neighbors of point i are given by
        neighbors1[indptr1[i]:indptr1[i+1]] and
        neighbors2[indptr2[i]:indptr2[i+1]]
    heap : the Fibonacci heap object to use
    nodes : the array of nodes to use
    """
    cdef unsigned int N = patch_indices.shape[0]
    cdef unsigned int MAX_POINTS = patch_indices.shape[1]
    cdef unsigned int count_popped = 0
    cdef unsigned int i_N
    cdef ITYPE_t i, idx2
    cdef FibonacciNode *v
    cdef FibonacciNode *current_neighbor
    cdef DTYPE_t dist, weight1, weight2

    # re-initialize nodes
    # children, parent, left_sibling, right_sibling should already be NULL
    # rank should already be 0, index will already be set
    # we just need to re-set state and val
    for i_N in range(0, N):
        nodes[i_N].state = 0  # 0 -> NOT_IN_HEAP
        nodes[i_N].val = 0
#    myset = set(theta0_v_idx)

    insert_node(heap, &nodes[i_node])

    while heap.min_node:
        v = remove_min(heap)
        v.state = 2  # 2 -> SCANNED
        # MaSIF: limit patches to MAX_POINTS(should be parameter)
        if count_popped >= MAX_POINTS: 
            continue

        for i from indptr1[v.index] <= i < indptr1[v.index + 1]:
            current_neighbor = &nodes[neighbors1[i]]
            if current_neighbor.state != 2:      # 2 -> SCANNED
                dist = distances1[i]
                if current_neighbor.state == 0:  # 0 -> NOT_IN_HEAP
                    current_neighbor.state = 1   # 1 -> IN_HEAP
                    current_neighbor.val = v.val + dist
                    insert_node(heap, current_neighbor)
#                    if current_neighbor.index in myset:
#                        idx2 = np.where(theta0_v_idx == current_neighbor.index)[0][0]
#                        current_neighbor.theta_val = theta0[idx2]
#                    else:
#                        current_neighbor.theta_val = v.theta_val
                else:
                    if current_neighbor.val > v.val + dist:
                        decrease_val(heap, current_neighbor,
                                 v.val + dist)
                    # MaSIF: if a neighbor is already in the queue, 
                            # update its theta angle value - simply average the angles.
#                    weight1 = (current_neighbor.val+v.val+dist) / current_neighbor.val
#                    weight2 = (current_neighbor.val+v.val+dist) / (v.val+dist)
#                    current_neighbor.theta_val = average_angles(current_neighbor.theta_val, v.theta_val, weight1, weight2)


        patch_indices[i_node][count_popped] = v.index
        patch_rho[i_node][count_popped] = v.val
        patch_theta[i_node][count_popped] = v.theta_val
        count_popped += 1


def dijkstra_entry(dist_matrix, theta0, theta0_v_idx, patch_indices, patch_rho, patch_theta):
    dijkstra(dist_matrix, patch_indices, patch_rho, patch_theta, theta0, theta0_v_idx)
    return patch_indices, patch_rho
