#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>

const int N=5;
const int cycles=5000; // # think/eat cycles

// for first version
std::vector<std::mutex> forks(N); // mutexes representing forks

// for second version
enum State { think, hungry, eat };
std::vector<State> state(N,think); // states of the philos
std::vector<std::condition_variable> cv(N); // for waiting
std::mutex mx; // mutex for protecting states
std::mutex px; // mutex for protecting cout

void philosopher1 (int i)
{
  int l=(i+1)%N; // right fork number
  int r=i; // left fork number
  for (int j=0; j<cycles; j++)
    {
      // think
      forks[l].lock();
      forks[r].lock();
      // eat
      forks[l].unlock();
      forks[r].unlock();      
    }
  std::lock_guard<std::mutex> lg{px};
  std::cout << "Philosopher " << i << " finished" << std::endl;
}

// thread i tests if thread j is able to eat
// thread i has the global lock mx
void test (int i, int j)
{
  int l=(j+1)%N;   // philosopher right from j
  int r=(j+N-1)%N; // philosopher left  from j
  if (state[j]==hungry && state[l]!=eat && state[r]!=eat)
    {
      state[j] = eat; // j can eat now
      if (j!=i) cv[j].notify_one();
    }
}

void philosopher2 (int i)
{
  int l=(i+1)%N; // right fork number
  int r=i; // left fork number
  for (int j=0; j<cycles; j++)
    {
      // think

      // I want to eat
      std::unique_lock<std::mutex> ul{mx}; // grab lock
      state[i] = hungry; // set state
      test(i,i); // check if I can eat
      // either my state is now eat or I wait until it becomes eat
      cv[i].wait(ul,[i]{return state[i]==eat;}); 
      ul.unlock();

      // eat

      // I am finished and may wake others
      ul.lock(); // grab lock
      state[i] = think; // set state
      test(i,(i+1)%N); // wake up left if possible
      test(i,(i+N-1)%N); // wake up right if possible
    }
  std::lock_guard<std::mutex> lg{px};
  std::cout << "Philosopher " << i << " finished" << std::endl;
}


int main ()
{
  std::vector<std::thread> ph1;
  for (int i=0; i<N; ++i)
    ph1.push_back(std::thread{philosopher2,i});
  for (int i=0; i<N; ++i)
    ph1[i].join();
}
