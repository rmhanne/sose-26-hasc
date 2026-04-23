#include <iostream>
#include <ostream>
#include <vector>
#include "time_experiment.hh"
//
// Created by robin-marcel on 4/23/26.
//
const long N = 1024 * 1024 * 512;
long acc = 0;

class EmptyLoop
{
public:
    void operator ()() const
    {
        for (long i = 0; i < N; i++)
        {
            // Wee waa wee woo we do nothing here wooo
        }
    }
};

class AccumulatingLoop
{
public:
    void operator ()() const
    {
        for (long i = 0; i < N; i++)
        {
           acc++;
        }
    }
};


std::vector<long> vec {};

class FillVectorLoop
{
public:
    void operator ()() const
    {
        vec.clear();
        for (long i = 0; i < N; i++)
        {
            vec.push_back(i);
        }
    }
};

int main(void)
{
    auto el = EmptyLoop();
    auto al = AccumulatingLoop();
    auto fl = FillVectorLoop();
    // Tests
    auto elr = time_experiment(el, 1);
    auto alr = time_experiment(al, 1);
    auto tlr = time_experiment(fl, 1);

    std::cout << "Empty loop ran " << elr.first <<  " times for a total of " << elr.second << " microsec. (" <<
        elr.second/elr.first << " per iteration)" << std::endl;
    std::cout << "Accumulating loop ran " << alr.first <<  " times for a total of " << alr.second << " microsec. (" <<
        alr.second/alr.first << " per iteration)" << std::endl;
    std::cout << "Fill Vector loop ran " << tlr.first <<  " times for a total of " << tlr.second << " microsec. (" <<
        tlr.second/tlr.first << " per iteration)" << std::endl;
}