#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  for (int i = 0; i < numNodes; ++i)
  {
    solution[i] = equal_prob;
  }

  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */

  double* score_old = new double[numNodes];
  double* score_new = new double[numNodes];

  std::copy(solution, solution + numNodes, score_old);
  std::fill(score_new, score_new + numNodes, 0.0);

  bool converged = false;

  while (!converged) {
      double no_outgoing_sum = 0.0;
      for (int v = 0; v < numNodes; v++) {
          if (outgoing_size(g, v) == 0) {
              no_outgoing_sum += (damping * score_old[v] / numNodes);
          }
      }

      for (int vi = 0; vi < numNodes; vi++) {
          const Vertex* start = incoming_begin(g, vi);
          const Vertex* end = incoming_end(g, vi);
          for (const Vertex* vj = start; vj != end; vj++) {
            if (outgoing_size(g, *vj) != 0) {
                score_new[vi] += (score_old[*vj] / outgoing_size(g, *vj));
            }
          }
          score_new[vi] = damping * score_new[vi] + (1.0 - damping) / numNodes;
          score_new[vi] += no_outgoing_sum;
      }

      double global_diff = 0.0;
      for (int vi = 0; vi < numNodes; vi++) {
          global_diff += std::abs(score_new[vi] - score_old[vi]);
          printf("globa diff: %f\n", global_diff);
      }
  
      converged = (global_diff < convergence);

      std::swap(score_old, score_new);
      std::fill(score_new, score_new + numNodes, 0.0);
  }

  std::copy(score_old, score_old + numNodes, solution);

  delete[] score_old;
  delete[] score_new;

}
