#include <iostream>
#include <vector>
#include <stdio.h>
#include <mpi.h>

int main (int argc, char** argv)
{

  // initialize mpi
  MPI_Init(&argc,&argv);

  // get rank and size
  int rank;
  int P;
  MPI_Comm_size(MPI_COMM_WORLD,&P);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  // check args
  if (argc!=3)
    {
      if (rank==0)
        std::cout << "usage: hello_mpi <size> <rounds>" << std::endl;
      MPI_Finalize();
      exit(0);
    }

  // read args
  int size; // message size
  int rounds;
  sscanf(argv[1],"%d",&size);
  sscanf(argv[2],"%d",&rounds);

  // prepare message
  std::vector<char> message(size,'A');
  int tag=50;

  for (int i=0; i<rounds; ++i)
    {
      if (rank==0) 
        {
          int dest = 1;
          MPI_Send(message.data(),message.size(),MPI_CHAR,
                   dest,tag,MPI_COMM_WORLD);
          int source = P-1;
          MPI_Status status;
          MPI_Recv(message.data(),message.size(),MPI_CHAR,source,tag,
                   MPI_COMM_WORLD,&status);
        }
      else 
        {
          int source = rank-1;
          MPI_Status status;
          MPI_Recv(message.data(),message.size(),MPI_CHAR,source,tag,
                   MPI_COMM_WORLD,&status);
          int dest = (rank+1)%P;
          MPI_Send(message.data(),message.size(),MPI_CHAR,
                   dest,tag,MPI_COMM_WORLD);
        }
    }

  // clean up mpi
  MPI_Finalize();
}
