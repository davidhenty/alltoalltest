#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

int main(void)
{
  MPI_Comm comm;

  double tstart, tstop, telapse;

  double *sbuf, *rbuf;

  int rank, size, i, irep, nbuf, nrep, ns, nr, irank, tag, bloop;

  MPI_Request *requests;
  MPI_Status *statuses;

  comm = MPI_COMM_WORLD;

  MPI_Init(NULL, NULL);

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  nbuf = 1;

  for (bloop=0; bloop < 10; bloop++)
    {
      ns = nbuf*size;
      nr = nbuf*size;
  
      if (rank == 0) printf("Running on %d processes with nbuf = %d\n", size, nbuf);

      sbuf = (double *) malloc(ns*sizeof(double));
      rbuf = (double *) malloc(nr*sizeof(double));

      requests = (MPI_Request *) malloc(2*size*sizeof(MPI_Request));
      statuses = (MPI_Status  *) malloc(2*size*sizeof(MPI_Status ));

      for (i=0; i < ns; i++)
        {
          sbuf[i] = (double) i;
        }

      for (i=0; i < nr; i++)
        {
          rbuf[i] = -1.0;
        }

      nrep = 100;
      tag = 0;

      MPI_Barrier(comm);

      tstart = MPI_Wtime();

      for (irep=0; irep < nrep; irep++)
        {
          MPI_Alltoall(sbuf, nbuf, MPI_DOUBLE, rbuf, nbuf, MPI_DOUBLE, comm);

          if (rank == size/2 && irep == nrep/2) printf("rbuf[3] = %g\n", rbuf[3]);
        }

      MPI_Barrier(comm);

      tstop = MPI_Wtime();

      telapse = tstop - tstart;

      if (rank == 0) printf("nbuf = %d: ave a2a time over %d reps was %g seconds\n", nbuf, nrep, telapse/((double) nrep));

      MPI_Barrier(comm);

      tstart = MPI_Wtime();

      for (irep=0; irep < nrep; irep++)
        {
          for (irank = 0; irank < size; irank++)
            {
              MPI_Irecv(&rbuf[nbuf*irank], nbuf, MPI_DOUBLE, irank, tag, comm, &requests[irank]);
            }

          for (irank = 0; irank < size; irank++)
            {
              MPI_Isend(&sbuf[nbuf*irank], nbuf, MPI_DOUBLE, irank, tag, comm, &requests[size+irank]);
            }
      
          MPI_Waitall(2*size, requests, statuses);

          if (rank == size/2 && irep == nrep/2) printf("rbuf[3] = %g\n", rbuf[3]);
        }

      MPI_Barrier(comm);

      tstop = MPI_Wtime();

      telapse = tstop - tstart;

      if (rank == 0) printf("nbuf = %d: ave p2p time over %d reps was %g seconds\n", nbuf, nrep, telapse/((double) nrep));

      free(sbuf);
      free(rbuf);
      free(requests);
      free(statuses);

      nbuf = 2*nbuf;
    }

  MPI_Finalize();

  return 0;
}
