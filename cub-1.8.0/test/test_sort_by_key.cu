#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <random>

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/util_allocator.cuh>

__global__ void kernel_sorted_by_key(int* g_keys, int* g_values, const int n){
    typedef cub::BlockRadixSort<int, 32, 4, int> BlockRadixSort;
    
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    int keys[4];
    int values[4];

    cub::LoadDirectBlocked(threadIdx.x, g_keys, keys);
    cub::LoadDirectBlocked(threadIdx.x, g_values, values);

    BlockRadixSort(temp_storage).Sort(keys, values);
    cub::StoreDirectBlocked(threadIdx.x, g_keys, keys); 
    cub::StoreDirectBlocked(threadIdx.x, g_values, values); 
}

void print_vec(std::vector<int>& data){
    for(int i = 0; i < data.size(); i++){
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}
int main(){
    const int n = 128;
    std::vector<int> keys(n), values(n);
    std::default_random_engine random(time(NULL));
    std::uniform_int_distribution<int> dis(0, 10);
    for(int i = 0; i < n; i++){
        keys[i] = dis(random);
        values[i] = i;
    }
    print_vec(keys);
    print_vec(values);

    int* g_keys, *g_values;
    cudaMalloc((void**)&g_keys, n * sizeof(int));
    cudaMalloc((void**)&g_values, n * sizeof(int));
    cudaMemcpy(g_keys, keys.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_values, values.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    kernel_sorted_by_key<<<1, 32>>>(g_keys, g_values, n);

    std::vector<int> h_keys(n), h_values(n);
    cudaMemcpy(h_keys.data(), g_keys, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_values.data(), g_values, n * sizeof(int), cudaMemcpyDeviceToHost);
    print_vec(h_keys);
    print_vec(h_values);
    return 0;
}
