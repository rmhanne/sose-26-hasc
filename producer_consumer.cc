#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

const int m=50;  // number of producers
const int n=10; // number of consumers
const int k=3;  // max number of tasks in queue
const int tasks=50; // # tasks produced per producer
using Task = std::pair<int,int>; // producer,task
std::mutex mx; // mutex for modification of queue
std::queue<Task> q; // the queue
std::condition_variable cvpush; // condition variable for push
std::condition_variable cvpop; // condition variable for pop
std::mutex fx; // mutex protecting finished
int finished=0; // number of finished producers
std::mutex px; // mutex for protecting cout

void producer (int i)
{
  // generate tasks
  for (int t=0; t<tasks; t++)
    {
      // grab lock for push access to queue
      std::unique_lock<std::mutex> ul{mx};
      // wait until queue has space available
      cvpush.wait(ul,[]{return q.size()<k;});
      // we have exclusivepush  access
      q.push(Task(i,t));
      // and notify a waiting thread (if any)
      cvpop.notify_one();
      // give away lock
      ul.unlock();
      // so that we can do something without
      // blocking others
      std::lock_guard<std::mutex> lg{px};
      std::cout << "Producer " << i
                << " queued task " << t
                << std::endl;
    }
  // count the number of finished producers
  fx.lock();
  finished += 1;
  // if I am the last one, wake all
  if (finished==m) cvpop.notify_all();
  fx.unlock();
  std::lock_guard<std::mutex> lg{px};
  std::cout << "Producer " << i << " finished" << std::endl;
}

void consumer (int j)
{
  // process tasks indefinitely
  while (1)
    {
      // grab lock for queue pop access
      std::unique_lock<std::mutex> ul{mx};
      // wait until there is a task or all are finished
      cvpop.wait(ul,[]{return q.size()>0 || finished==m;});
      // we are here, when
      // q.size()>0 && finished<m -> there is a task and more are coming
      // q.size()==0 && finished==m -> exit, we are done
      // q.size()>0 && finished==m -> there are more tasks
      // if there are no tasks and all are finished we are done
      if (q.size()==0 && finished==m)
        break;
      // so there is at least one task, get it
      Task task = q.front();
      q.pop();
      cvpush.notify_one();
      ul.unlock();
      std::lock_guard<std::mutex> lg{px};
      std::cout << "Consumer " << j << " processing task "
                << task.second << " from "
                << task.first << std::endl;
    }
  std::lock_guard<std::mutex> lg{px};
  std::cout << "Consumer " << j << " finished" << std::endl;
}

int main ()
{
  std::vector<std::thread> producers;
  for (int i=0; i<m; ++i)
    producers.push_back(std::thread{producer,i});
  std::vector<std::thread> consumers;
  for (int i=0; i<n; ++i)
    consumers.push_back(std::thread{consumer,i});
  for (int i=0; i<m; ++i)
    producers[i].join();
  for (int i=0; i<n; ++i)
    consumers[i].join();
}
