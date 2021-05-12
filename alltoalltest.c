#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

int main(void)
{
  MPI_Comm comm;

  double tstart, tstop, telapse;

  double *sbuf, *rbufa, *rbufp;

  int rank, size, i, irep, nmsg, nrep, nbuf, irank, tag, bloop;

  MPI_Request *requests;
  MPI_Status *statuses;

  comm = MPI_COMM_WORLD;

  MPI_Init(NULL, NULL);

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  nmsg = 1;

  for (bloop=0; bloop < 1; bloop++)
    {
      nbuf = nmsg*size;
  
      if (rank == 0) printf("Running on %d processes with nmsg = %d\n", size, nmsg);

      sbuf =  (double *) malloc(nbuf*sizeof(double));
      rbufa = (double *) malloc(nbuf*sizeof(double));
      rbufp = (double *) malloc(nbuf*sizeof(double));

      requests = (MPI_Request *) malloc(2*size*sizeof(MPI_Request));
      statuses = (MPI_Status  *) malloc(2*size*sizeof(MPI_Status ));

      for (i=0; i < nbuf; i++)
        {
          sbuf[i] = rank*nbuf + (double) i;
        }

      for (i=0; i < nbuf; i++)
        {
          rbufa[i] = -1.0;
          rbufp[i] = -1.0;
        }

      nrep = 100;
      tag = 0;

      MPI_Barrier(comm);

      tstart = MPI_Wtime();

      for (irep=0; irep < nrep; irep++)
        {
          MPI_Alltoall(sbuf, nmsg, MPI_DOUBLE, rbufa, nmsg, MPI_DOUBLE, comm);

          if (rank == size && irep == nrep/2) printf("rbufa[3] = %g\n", rbufa[3]);
        }

      MPI_Barrier(comm);

      tstop = MPI_Wtime();

      telapse = tstop - tstart;

      if (rank == 0) printf("nmsg = %d: ave a2a time over %d reps was %g seconds\n", nmsg, nrep, telapse/((double) nrep));

      MPI_Barrier(comm);

      tstart = MPI_Wtime();

      for (irep=0; irep < nrep; irep++)
        {
          for (irank = 0; irank < size; irank++)
            {
              MPI_Irecv(&rbufp[nmsg*irank], nmsg, MPI_DOUBLE, irank, tag, comm, &requests[irank]);
            }

          for (irank = 0; irank < size; irank++)
            {
              MPI_Isend(&sbuf[nmsg*irank], nmsg, MPI_DOUBLE, irank, tag, comm, &requests[size+irank]);
            }
      
          MPI_Waitall(2*size, requests, statuses);

          if (rank == size && irep == nrep/2) printf("rbufp[3] = %g\n", rbufp[3]);
        }

      MPI_Barrier(comm);

      tstop = MPI_Wtime();

      telapse = tstop - tstart;

      if (rank == 0) printf("nmsg = %d: ave p2p time over %d reps was %g seconds\n", nmsg, nrep, telapse/((double) nrep));

      for (i=0; i < nbuf; i++)
        {
          if (rbufp[i] != rbufa[i])
            {
              printf("ERROR on rank %d: rbufp[%d] = %g != rbufa[%d] = %g\n", rank, i, rbufp[i], i, rbufa[i]);
              break;
            }
        }

      // node-aware by-hand

      MPI_Comm nodecomm, spancomm;
      int nodesize, numnode, noderank, nodenum;

      char nodename[MPI_MAX_PROCESSOR_NAME];
      int iblk, jblk, mblk, nblk;

      double *tbuf1, *tbuf2; // temporaries, one per node

      // Create node-local communicators

      MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank,
                          MPI_INFO_NULL, &nodecomm);

      MPI_Comm_rank(nodecomm, &noderank);
      MPI_Comm_size(nodecomm, &nodesize);

      // Create "spanning" communicators with colour and key = noderank

      MPI_Comm_split(comm, noderank, noderank, &spancomm);

      MPI_Comm_rank(spancomm, &nodenum);
      MPI_Comm_size(spancomm, &numnode);

      MPI_Get_processor_name(nodename, &irank);

      if (rank == 0) printf("Running on %d nodes\n", numnode);

      if (noderank == 0) printf("Node %d is <%s> with %d procs\n",
                                nodenum, nodename, nodesize);
      
      if (noderank == 0)
        {
          tbuf1 = (double *) malloc(nodesize*nbuf*sizeof(double));
          tbuf2 = (double *) malloc(nodesize*nbuf*sizeof(double));
        }

      for (i=0; i < nbuf; i++)
        {
          rbufp[i] = -1.0;
        }

      MPI_Barrier(comm);

      tstart = MPI_Wtime();

      // gather on a node

      printf("Gathering ... on rank %d\n", rank);

      MPI_Gather(sbuf,  nbuf*nodesize, MPI_DOUBLE,
                 tbuf1, nbuf*nodesize, MPI_DOUBLE, 0, nodecomm);

      printf("... done on rank %d\n", rank);

      // pack the data - could maybe avoid with clever use of datatypes

      if (noderank == 0)
        {
          for (i=0; i < nbuf*nodesize; i++)
            {
              printf("Bef: nodenum, %d: tbuf1[%d] = %g, tbuf2[%d] = %g\n",
                     nodenum, i, tbuf1[i], i, tbuf2[i]);
            }

          printf("transposing on node %d ...\n", nodenum);

          // Transpose from a size/numnode x size matrix to a size x
          // size/numnode matrix. Each "element" has nmsg entries.

          // Loop over entries in the source matrix in order
          // Source is of size mblk x nblk

          mblk = size/numnode;
          nblk = size;

          for (iblk=0; iblk < mblk; iblk++)
            {
              for (jblk = 0; jblk < nblk; jblk++)
                {
                  for (i=0; i < nmsg; i++)
                    {
                      {
                        tbuf2[(jblk*mblk + iblk)*nmsg + i] = \
                        tbuf1[(iblk*nblk + jblk)*nmsg+i];
                      }
                    }
                }
            }

          printf("... done on node %d\n", nodenum);

          for (i=0; i < nbuf*nodesize; i++)
            {
              printf("Aft: nodenum %d: tbuf1[%d] = %g, tbuf2[%d] = %g\n",
                     nodenum, i, tbuf1[i], i, tbuf2[i]);
            }

          // Now do the alltoall between nodes in spancomm

          //          MPI_Alltoall(tbuf2, MPI_DOUBLE, nbuf,
          //tbuf1, MPI_DOUBLE, nbuf,
          //spancomm);
        }

      MPI_Barrier(comm);

      tstop = MPI_Wtime();

      telapse = tstop - tstart;

      if (rank == 0) printf("nmsg = %d: ave p2p time over %d reps was %g seconds\n", nmsg, nrep, telapse/((double) nrep));

      for (i=0; i < nbuf; i++)
        {
          if (rbufp[i] != rbufa[i])
            {
              printf("ERROR on rank %d: rbufp[%d] = %g != rbufa[%d] = %g\n", rank, i, rbufp[i], i, rbufa[i]);
              break;
            }
        }

      free(sbuf);
      free(rbufa);
      free(rbufp);
      free(requests);
      free(statuses);

      nmsg = 2*nmsg;
    }

  MPI_Finalize();

  return 0;
}
