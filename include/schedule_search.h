#ifndef SCHEDULE_SEARCH_H
#define SCHEDULE_SEARCH_H

#include <cstddef>
#include <string>
#include <vector>

#include "ttl/Tensor.h"

struct ConvWorkload {
  int batch;
  int height;
  int width;
  int channels;
  int filters;
  int kernel;
  int stride;
  int pad;
};

struct ConvSchedule {
  int tileH;
  int tileW;
  int channelBlock;
  int numThreads;
  bool kernelOuter;
};

struct SearchConfig {
  int warmupIters;
  int benchmarkIters;
  int randomBudget;
  int hillSteps;
  int maxThreads;
  int gridMaxEvals;
};

struct SearchResult {
  std::string technique;
  ConvSchedule bestSchedule;
  double bestMs;
  std::size_t evaluatedCount;
};

std::string schedule_to_string(const ConvSchedule& schedule);

ConvWorkload workload_from_name(const std::string& name);

std::vector<ConvSchedule> build_candidate_pool(int maxThreads);

double evaluate_schedule(const ConvWorkload& workload,
                         const ConvSchedule& schedule,
                         int warmupIters,
                         int benchmarkIters);

SearchResult run_grid_search(const ConvWorkload& workload,
                             const SearchConfig& config,
                             const std::vector<ConvSchedule>& pool);

SearchResult run_random_search(const ConvWorkload& workload,
                               const SearchConfig& config,
                               const std::vector<ConvSchedule>& pool,
                               unsigned int seed);

SearchResult run_hill_climb_search(const ConvWorkload& workload,
                                   const SearchConfig& config,
                                   const std::vector<ConvSchedule>& pool,
                                   unsigned int seed);

#endif
