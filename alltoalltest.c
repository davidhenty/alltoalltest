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

  MPI_Comm nodecomm, spancomm;
  int nodesize, numnode, noderank, nodenum;

  char nodename[MPI_MAX_PROCESSOR_NAME];
  int iblk, jblk, mblk, nblk;

  double *tbuf1, *tbuf2; // temporaries, one per node

  MPI_Datatype vector, resizevector;
  MPI_Aint dblesize;

  comm = MPI_COMM_WORLD;

  MPI_Init(NULL, NULL);

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // node-aware by-hand

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
      
  nmsg = 1;

  for (bloop=0; bloop < 10; bloop++)
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
          sbuf[i] = rank*nbuf + (double) (i+1);
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

      if (noderank == 0)
        {
          tbuf1 = (double *) malloc(nodesize*nbuf*sizeof(double));
          tbuf2 = (double *) malloc(nodesize*nbuf*sizeof(double));
        }

      for (i=0; i < nbuf; i++)
        {
          rbufp[i] = -1.0;
        }

      // For simplicity assume nmsg = 1 for this argument. After the
      // gather and alltoall betwee nodes, rank 0 on each node node
      // now has a matrix of size nblk x mblk, made up of numnode mats
      // of size mblk x mblk tiled vertically (thinking C storage
      // order here) Need to send out sections of size mblk separated
      // by a stride of mbl*mblk. Count is numnode. Total amount of
      // data to each process is numnode * mblk = numnode * nodesize =
      // size as required.

      MPI_Type_vector(numnode, nodesize*nmsg, nodesize*nodesize*nmsg,
                      MPI_DOUBLE, &vector);

      // Need to resize it to be nodesize*msg doubles so it tiles
      // correctly.

      MPI_Type_extent(MPI_DOUBLE, &dblesize);

      MPI_Type_create_resized(vector, 0,
                              nodesize*nmsg*dblesize, &resizevector);
      
      MPI_Type_commit(&resizevector);

      MPI_Barrier(comm);

      tstart = MPI_Wtime();

      for (irep=0; irep < nrep; irep++)
        {
          // gather on a node

          //      printf("Gathering ... on rank %d\n", rank);

          MPI_Gather(sbuf,  nbuf, MPI_DOUBLE,
                     tbuf1, nbuf, MPI_DOUBLE, 0, nodecomm);

          //      printf("... done on rank %d\n", rank);

          // pack the data - could maybe avoid with clever use of datatypes

          if (noderank == 0)
            {
              // Transpose from a size/numnode x size matrix to a size x
              // size/numnode matrix. Each "element" has nmsg entries.

              // Loop over entries in the source matrix in order
              // Source is of size mblk x nblk

              mblk = nodesize;  // = size/numnodes
              nblk = size;

              for (iblk=0; iblk < mblk; iblk++)
                {
                  for (jblk = 0; jblk < nblk; jblk++)
                    {
                      for (i=0; i < nmsg; i++)
                        {
                          {
                            tbuf2[(jblk*mblk + iblk)*nmsg + i] =
                            tbuf1[(iblk*nblk + jblk)*nmsg + i]   ;
                          }
                        }
                    }
                }

              // Now do the alltoall between nodes in spancomm. Try not to
              // use derived types here as it is the most costly operation
              // so do not want to slow it down.
              // Amount of data to send to each node is:
              // total per node/numnode = nmsg*size*nodesize/numnode
              // = nmsg*nodesize*nodesize

              MPI_Alltoall(tbuf2, nmsg*nodesize*nodesize, MPI_DOUBLE,
                           tbuf1, nmsg*nodesize*nodesize, MPI_DOUBLE,
                           spancomm);
            }

          // Now scatter out across each node. Need a derived type for
          // sendtype as data received contiguously from a single source
          // node is scattered across different ranks on this node.

          MPI_Scatter(tbuf1, 1, resizevector,
                      rbufp, nbuf, MPI_DOUBLE, 0, nodecomm);

        }

      MPI_Barrier(comm);
      
      tstop = MPI_Wtime();

      telapse = tstop - tstart;

      if (rank == 0) printf("nmsg = %d: ave dsh time over %d reps was %g seconds\n", nmsg, nrep, telapse/((double) nrep));

      for (i=0; i < nbuf; i++)
        {
          if (rbufp[i] != rbufa[i])
            {
              printf("ERROR on rank %d: rbufp[%d] = %g != rbufa[%d] = %g\n", rank, i, rbufp[i], i, rbufa[i]);
              break;
            }
        }

      if (noderank == 0)
        {
          free(tbuf1);
          free(tbuf2);
        }

      MPI_Type_free(&resizevector);
      MPI_Type_free(&vector);

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
