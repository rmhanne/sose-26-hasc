#include <iostream>
#include <thread>
#include <atomic>

int in1=0;   // indicates that thread 1 is in critical section
int in2=0;   // indicates that thread 2 is in critical section
int last=1;  // process that arrived last at critical section
int count=0; // counts how many times the critical section was passed
int n=30000000; // how many times ech process passes
std::atomic<int> x{0};
std::atomic_flag f{ATOMIC_FLAG_INIT};

void P1_nolock ()
{
  for (int i=0; i<n; i++)
    count += 1;
}

void P2_nolock ()
{
  for (int i=0; i<n; i++)
    count += 1;
}

void P1_peterson ()
{
  for (int i=0; i<n; i++)
    {
      in1 = 1;     // I want to enter
      last = 1;    // tie breaker
      while (in2 && last==1) ; // busy wait
      count += 1;  // increment counter in c.s.
      in1 = 0;     // we are done
    }
}

void P2_peterson ()
{
  for (int i=0; i<n; i++)
    {
      in2 = 1;     // I want to enter
      last = 2;    // tie breaker
      while (in1 && last==2) ; // busy wait
      count += 1;  // increment counter in c.s.
      in2 = 0;     // we are done
    }
}

void P1_peterson_fenced ()
{
  for (int i=0; i<n; i++)
    {
      in1 = 1;     // I want to enter
      last = 1;    // tie breaker
      std::atomic_thread_fence(std::memory_order_seq_cst);
      while (in2 && last==1)
        std::atomic_thread_fence(std::memory_order_seq_cst);
      count += 1;  // increment counter in c.s.
      in1 = 0;     // we are done
    }
}

void P2_peterson_fenced ()
{
  for (int i=0; i<n; i++)
    {
      in2 = 1;     // I want to enter
      last = 2;    // tie breaker
      std::atomic_thread_fence(std::memory_order_seq_cst);
      while (in1 && last==2)
        std::atomic_thread_fence(std::memory_order_seq_cst);
      count += 1;  // increment counter in c.s.
      in2 = 0;     // we are done
    }
}

void P1_atomic_flag ()
{
  for (int i=0; i<n; i++)
    {
      bool z = f.test_and_set();
      while (z) z = f.test_and_set();
      count += 1;  // increment counter in c.s.
      f.clear();   // we are done
    }
}

void P2_atomic_flag ()
{
  for (int i=0; i<n; i++)
    {
      bool z = f.test_and_set();
      while (z) z = f.test_and_set();
      count += 1;  // increment counter in c.s.
      f.clear();   // we are done
    }
}

void P1_atomic ()
{
  for (int i=0; i<n; i++)
    {
      int z = x.fetch_add(1);
      while (z>0) {
        x.fetch_sub(1);
        z = x.fetch_add(1);
      }
      count += 1;  // increment counter in c.s.
      x.fetch_sub(1);   // we are done
    }
}

void P2_atomic ()
{
  for (int i=0; i<n; i++)
    {
      int z = x.fetch_add(1);
      while (z>0) {
        x.fetch_sub(1);
        z = x.fetch_add(1);
      }
      count += 1;  // increment counter in c.s.
      x.fetch_sub(1);   // we are done
    }
}

int main ()
{
  std::thread t1a{P1_nolock};
  std::thread t2a{P2_nolock};
  t1a.join();
  t2a.join();
  std::cout << "no_lock        : count is " << count << " and should be " << 2*n << std::endl;
  count=0;
  std::thread t1b{P1_peterson};
  std::thread t2b{P2_peterson};
  t1b.join();
  t2b.join();
  std::cout << "peterson       : count is " << count << " and should be " << 2*n << std::endl;
  count=0;
  std::thread t1c{P1_peterson_fenced};
  std::thread t2c{P2_peterson_fenced};
  t1c.join();
  t2c.join();
  std::cout << "peterson_fenced: count is " << count << " and should be " << 2*n << std::endl;
  count=0;
  std::thread t1d{P1_atomic_flag};
  std::thread t2d{P2_atomic_flag};
  t1d.join();
  t2d.join();
  std::cout << "peterson_atomic_flag: count is " << count << " and should be " << 2*n << std::endl;
  count=0;
  std::thread t1e{P1_atomic};
  std::thread t2e{P2_atomic};
  t1e.join();
  t2e.join();
  std::cout << "peterson_atomic: count is " << count << " and should be " << 2*n << std::endl;
}
