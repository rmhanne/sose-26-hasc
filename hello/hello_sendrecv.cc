#include <iostream>
#include <thread>

#include "MessageSystem.hh"

void thread (const int rank, const int P)
{
  register_thread(rank);
  int x = rank;

  // blocking sends
  int y = -1;
  if (rank==0)
    {
      send(1,x);
      recv(1,y);
    }
  else
    {
      recv(0,y);
      send(0,x);
    }
  std::cout << rank << ": y=" << y << std::endl;

  // rprobe and recvany
  y = -1;
  if (rank==0)
    {
      send(1,x);
      while (!rprobe(1)) ;
      recv(1,y);
    }
  else
    {
      int from;
      recvany(from,y);
      send(0,x);
    }
  std::cout << rank << ": y=" << y << std::endl;

  // nonblocking send/recv
  y = -1;
  auto msg1 = asend(1-rank,x);
  auto msg2 = arecv(1-rank,y);
  // need to wait for all outstanding receives
  while (!test<int>(msg2)) ;
  // before waiting for the sends!!
  while (!test<int>(msg1)) ;
  std::cout << rank << ": y=" << y << std::endl;
}

int main (int argc, char** argv)
{
  initialize_message_system(2);
  std::thread t1{thread,0,2};
  std::thread t2{thread,1,2};
  t1.join();
  t2.join();
}
