#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1


void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{
    #pragma omp parallel for
    for (int i = 0; i < frontier->count; i++)
    {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];

            if (distances[outgoing] == NOT_VISITED_MARKER)
            {
                distances[outgoing] = distances[node] + 1;
                int old_count;
                do {
                    old_count = new_frontier->count;
                } while (!__sync_bool_compare_and_swap(&new_frontier->count, old_count, old_count + 1));

                if (old_count < new_frontier->max_vertices) {
                    new_frontier->vertices[old_count] = outgoing;
                } else {
                    printf("Error: new_frontier is full\n");
                    exit(1);
                }
            }
        }
    }
}

void bottom_up_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances,
    int depth
)
{
    #pragma omp parallel for
    for (int v=0; v < g->num_nodes; v++)
    {
        // If the vertex has not been visited
        if (distances[v] == NOT_VISITED_MARKER)
        {
            int start_edge = g->incoming_starts[v];
            int end_edge = (v == g->num_nodes - 1)
                               ? g->num_edges
                               : g->incoming_starts[v + 1];

            // Check if any of the neighbors are in the frontier
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int incoming = g->incoming_edges[neighbor];

                // If the vertex shares an incoming edge with a vertex on the frontier
                if (distances[incoming] == depth)
                {
                    distances[v] = distances[incoming] + 1;
                    int old_count = __sync_fetch_and_add(&new_frontier->count, 1);
                    if (old_count < new_frontier->max_vertices) {
                        new_frontier->vertices[old_count] = v;
                        break;
                    } else {
                        break;
                    }
                }
            }
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set list2;
    vertex_set_init(&list2, graph->num_nodes);
    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // Initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // Setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int depth = 0;
    // While there are vertices in the frontier
    while (frontier->count != 0)
    {
        // Clear the new frontier
        vertex_set_clear(new_frontier);

        // Perform the bottom-up BFS step
        bottom_up_step(graph, frontier, new_frontier, sol->distances, depth++);

        // Swap the frontier and the new frontier
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}



void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set list2;
    vertex_set_init(&list2, graph->num_nodes);
    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;
    int THRESHOLD = graph->num_nodes / omp_get_max_threads();
    int depth = 0;

    // Initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // Setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    // While there are vertices in the frontier
    while (frontier->count != 0)
    {
        // Clear the new frontier
        vertex_set_clear(new_frontier);

        // If the size of the frontier is less than a threshold, use the top-down approach
        if (frontier->count < THRESHOLD)
            top_down_step(graph, frontier, new_frontier, sol->distances);
        // Otherwise, use the bottom-up approach
        else
            bottom_up_step(graph, frontier, new_frontier, sol->distances, depth);

        // Swap the frontier and the new frontier
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        depth++;
    }
}
