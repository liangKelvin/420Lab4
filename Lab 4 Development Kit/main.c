	
#define LAB4_EXTEND

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Lab4_IO.h"
#include "timer.h"
#include "mpi.h"

#define EPSILON 0.00001
#define DAMPING_FACTOR 0.85
#define THRESHOLD 0.0001

int main (int argc, char* argv[]){

	struct node *nodehead;
	int nodecount;
	int *num_in_links, *num_out_links;
	double *r, *r_pre;
	double *r_local;
	int i, j;
	double damp_const;
	int iterationcount = 0;
	int collected_nodecount;
	double start;
	double end;
	int myRank;
	int size;

	MPI_Init(NULL, NULL);

	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (get_node_stat(&nodecount, &num_in_links, &num_out_links)) return 254;
	
	int lowBound = nodecount*myRank / size;
	int highBound = (myRank+1)*nodecount / size;
	int nodecount_local = nodecount / size; 

	// Calculate the result
	if (node_init(&nodehead, num_in_links, num_out_links, lowBound, highBound)) return 254;
	
	r = malloc(nodecount * sizeof(double));
	r_pre = malloc(nodecount * sizeof(double));
	r_local = malloc(nodecount_local * sizeof(double));
	for ( i = 0; i < nodecount; ++i)
		r[i] = 1.0 / nodecount;
	damp_const = (1.0 - DAMPING_FACTOR) / nodecount;
   

	GET_TIME(start);
	// CORE CALCULATION
	do{
		++iterationcount;
		vec_cp(r, r_pre, nodecount);
		for ( i = 0; i < nodecount_local; ++i){
			r_local[i] = 0;
			for ( j = 0; j < nodehead[i].num_in_links; ++j)
				r_local[i] += r_pre[nodehead[i].inlinks[j]] / num_out_links[nodehead[i].inlinks[j]];
			r_local[i] *= DAMPING_FACTOR;
			r_local[i] += damp_const;
		}
		MPI_Allgather(r_local, nodecount_local, MPI_DOUBLE, r, nodecount_local, MPI_DOUBLE, MPI_COMM_WORLD);
	}while(rel_error(r, r_pre, nodecount) >= EPSILON);
	//printf("Program converges at %d th iteration.\n", iterationcount);
	 


	MPI_Finalize();

	GET_TIME(end);

	if(myRank == 0) {
		Lab4_saveoutput(r, nodecount, end-start);
	}

	// post processing
	node_destroy(nodehead, nodecount);
	free(num_in_links); free(num_out_links);

	return 0;
}


