#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>

const int N=5;
const int cycles=3; // # runs through barrier

int count=0; // count number of threads that arrived at the barrier
std::vector<int> flag(N,0); // flag indicating waiting thread
std::mutex mx; // mutex for use with the cvs
std::vector<std::condition_variable> cv(N); // for waiting
std::mutex px; // mutex for protecting cout

void barrier (int i)
{
  std::unique_lock<std::mutex> ul{mx};
  count += 1; // one more
  if (count<N)
    {
      // wait on my cv until all have arrived
      flag[i] = 1; // indicate I am waiting
      cv[i].wait(ul,[i]{return flag[i]==0;}); // wait
    }
  else
    {
      // I am the last one, lets wake them up
      count = 0; // reset counter for next turn
      std::cout << "------" << std::endl;
      for (int j=0; j<N; j++)
	if (flag[j]==1)
	  {
	    flag[j] = 0; // the event
	    cv[j].notify_one(); // wake up
	  }
    }
}

void f (int i)
{
  for (int j=0; j<cycles; j++)
    {
      barrier(i); // block until all threads arrived
      std::lock_guard<std::mutex> lg{px};
      std::cout << "Thread " << i << std::endl;
    }
}

int main ()
{
  std::vector<std::thread> th;
  for (int i=0; i<N; ++i)
    th.push_back(std::thread{f,i});
  for (int i=0; i<N; ++i)
    th[i].join();
}
