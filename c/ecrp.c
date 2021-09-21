#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

int uncat(unsigned int * r_state, double * weights, int n, double alpha)
{
    int i = 0;
    double r = (n + 1 + alpha) * rand_r(r_state) / ((long) RAND_MAX + 1);
    r -= weights[i];
    while (r > 0)
    {
        ++i;
        r -= weights[i];
    }
    return i;
}

double getEntropy(double * p, int n)
{
    double entropy = 0.0;
    int idx = 0;
    while (p[idx] > 0)
    {
        entropy -= log2(p[idx] /  n) * p[idx] /  n;
        ++idx;
    }
    return entropy;
}

const int arrSize = 0x1000;

double ecrp(double alpha, int beta, int steps, unsigned int * r_state)
{
    double weights[arrSize];
    for (int i = 0; i < arrSize; ++i)
        weights[i] = 0;
    weights[0] = 1.0;
    int addendIdxs[1000];
    for (int i = 0; i < steps; ++i)
    {
        int nzIdx = 0;
        while (weights[nzIdx] > 0.0)
            ++nzIdx;
        weights[nzIdx] = alpha;
        for (int j = 0; j < beta; ++j)
        {
            addendIdxs[j] = uncat(r_state, weights, i, alpha);
        }
        for (int j = 0; j < beta; ++j)
            weights[addendIdxs[j]] += 1.0 / beta;
        weights[nzIdx] -= alpha;
    }
    double entropy =  getEntropy(weights, steps);
    return entropy;
}


int main()
{
    srand(time(NULL));

    const double alpha = 5.0;
    const int steps = 1000;
    const int n = 10000;
    const double lo = 0;
    const double hi = 3;

    unsigned int seeds[n];
    for (int i = 0; i < n; ++i)
        seeds[i] = rand();
    double xs[n];
    double ys[n];

    #pragma omp parallel
    {
        int threadnum = omp_get_thread_num();
        int numthreads = omp_get_num_threads();
        for (int i = threadnum; i < n; i += numthreads)
        {
            int beta = (int) pow(10, lo + i * (hi - lo) / (n - 1));
            xs[i] = pow(10, lo + i * (hi - lo) / (n - 1));
            ys[i] = ecrp(alpha, beta, steps, &seeds[i]);
        }
    }
    for (int i = 0; i < n; ++i)
        printf("%f,%f\n", xs[i], ys[i]);
    
    return 0;
}
