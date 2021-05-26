#include <iostream>
#include <thread>

#include "MessageSystem.hh"

void thread (const int rank, const int P)
{
  register_thread(rank);
  int x = rank;
  int y = -1;
  if (rank==0)
    {
      send(1,x);
      recv(1,y);
      break;
    }
  else
    {
      recv(0,y);
      send(0,x);
    }
  std::cout << rank << ": y=" << y << std::endl;
}

int main (int argc, char** argv)
{
  ms = std::make_shared<MessageSystem>(2);
  std::thread t1{thread,0,2};
  std::thread t2{thread,1,2};
  t1.join();
  t2.join();
}
