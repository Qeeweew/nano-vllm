#pragma once
#include <ATen/Parallel.h>

enum class ExecutionPolicy { Parallel, Sequential };

template <ExecutionPolicy Policy, typename F>
inline void dispatch_for(int64_t begin, int64_t end, F&& func) {
    if constexpr (Policy == ExecutionPolicy::Parallel) {
        at::parallel_for(begin, end, 0, [&](int64_t start_chunk, int64_t end_chunk) {
            for (int64_t i = start_chunk; i < end_chunk; ++i) {
                func(i); // Execute the lambda body for each index in the chunk
            }
        });
    } else { // Sequential
        for (int64_t i = begin; i < end; ++i) {
            func(i); // Execute the lambda body directly
        }
    }
}