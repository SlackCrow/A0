#ifndef A0_HPP
#define A0_HPP
#include <functional> 
#include <math.h>

template <typename T, typename Op>
void omp_scan(int n, const T* in, T* out, Op op) 
{
    // # of total threads
    int maxNThreads = omp_get_max_threads();  

    // local scan block
    std::vector<T> localSum(maxNThreads);
    std::vector<T> scannedBlockSum;

    // find the nearest n so that n%p would be 0 
    int rem = n%maxNThreads;
    int sizeOfN; 
    if(rem == 0)
    {
        sizeOfN = n; 
    }
    else
    {
        sizeOfN =  (n - rem) + maxNThreads;
    }

    int arraySize = sizeOfN / maxNThreads; // array size per thread
    
    #pragma omp parallel default(none) shared(n,in, out, op, arraySize, localSum, scannedBlockSum, sizeOfN, maxNThreads) // share variables needed
    {
        // # of threads available
        int currentThreadId = omp_get_thread_num();
        T* temp = new T[arraySize]; // allocate a temp array in the heap, as this can get bigger than the stack
        int start = currentThreadId*arraySize;
        int end = ((currentThreadId*arraySize)+arraySize);
        
        // split the input array
        #pragma omp simd // only works with AVX2 on really because of the instruction needed to calculate dependencies
        for(int j=start; j<end;j++) 
        {
            if(j >= n) // if j is bigger than or equal to  n, pad it by 0 
            {
                temp[j-start] = 0;
            }
            else
            {
                temp[j-start] = in[j];
            }
        }               
        std::vector<T> current(arraySize, 1);
        partial_sum(temp, temp+arraySize, current.begin(), op); // calculate partial sum of a local block
        delete [] temp; // delete temp
        localSum[currentThreadId] = current.back(); // get the sum of the local block
        
        #pragma omp barrier // calculate prefix sum of block sums
        {
            #pragma omp single // only one thread calculates this part
            {
                std::vector<T> toRet(localSum.size(), 1);
                partial_sum(localSum.begin(), localSum.end(), toRet.begin(), op); 
                toRet.push_back(op(toRet.back(),localSum[0]));
                scannedBlockSum = toRet; 
                #pragma omp flush
            }
        }
        #pragma omp for
        for(int i=0; i<maxNThreads; i++)
        {
            int currentThreadId = i;
            if(arraySize == 1) // if there is only one element per PE copy the scannedBlockSum
            {
                std::copy(scannedBlockSum.begin(),scannedBlockSum.begin()+scannedBlockSum.size(),out+scannedBlockSum.size()*currentThreadId); // copy the data from a part of localsum to out
            }
            else
            {
                /* code */
                if(currentThreadId > 0 ) // skip the first block
                {
                    // increment each block by the block sum
                    std::transform(current.begin(), current.end(), current.begin(), std::bind(op, std::placeholders::_1, scannedBlockSum[currentThreadId-1]));
                }
                if(currentThreadId == maxNThreads-1 && n > 90) // check if it's the last thread or not
                {
                    // For some reason Clang is totally fine with copying beyond the boundary but GCC is not, so throw away the data that goes beyond the boundary of out. 
                    int throwAway = (current.size() * maxNThreads) - n; 
                    std::copy(current.begin(),current.begin()+(current.size()-throwAway),out+current.size()*currentThreadId); // copy to the output array 
                }
                else
                {
                    std::copy(current.begin(),current.begin()+current.size(),out+current.size()*currentThreadId); // copy the data from a part of localsum to out
                }
            }
        #pragma omp flush
        }
    }
} // omp_scan

#endif // A0_HPP
