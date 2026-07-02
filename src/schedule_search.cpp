#include "schedule_search.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>

namespace {

using Tensor4f = ttl::Tensor<4, float>;

struct ConvContext {
  ConvWorkload workload;
  Tensor4f input;
  Tensor4f weights;
  Tensor4f output;

  explicit ConvContext(const ConvWorkload& w)
      : workload(w),
        input({w.batch, w.height + 2 * w.pad, w.width + 2 * w.pad, w.channels}),
        weights({w.kernel, w.kernel, w.channels, w.filters}),
        output({w.batch,
                (w.height + 2 * w.pad - w.kernel) / w.stride + 1,
                (w.width + 2 * w.pad - w.kernel) / w.stride + 1,
                w.filters}) {}
};

void fill_tensor(Tensor4f& tensor, float scale, float bias) {
  for (int i = 0; i < tensor.getShape().size(); ++i) {
    const float value = static_cast<float>((i % 131) - 65) / 65.0f;
    tensor[i] = scale * value + bias;
  }
}

void run_conv_schedule(ConvContext& ctx, const ConvSchedule& schedule) {
  const int outH = ctx.output.getShape()[1];
  const int outW = ctx.output.getShape()[2];

  const int tilesH = (outH + schedule.tileH - 1) / schedule.tileH;
  const int tilesW = (outW + schedule.tileW - 1) / schedule.tileW;
  const int totalTiles = ctx.workload.batch * ctx.workload.filters * tilesH * tilesW;

  const int threads = std::max(1, schedule.numThreads);
  std::atomic<int> nextTile{0};
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(threads));

  auto worker = [&]() {
    while (true) {
      const int tileIndex = nextTile.fetch_add(1, std::memory_order_relaxed);
      if (tileIndex >= totalTiles) {
        return;
      }

      int rem = tileIndex;
      const int b = rem / (ctx.workload.filters * tilesH * tilesW);
      rem %= (ctx.workload.filters * tilesH * tilesW);
      const int k = rem / (tilesH * tilesW);
      rem %= (tilesH * tilesW);
      const int th = rem / tilesW;
      const int tw = rem % tilesW;

      const int hStart = th * schedule.tileH;
      const int hEnd = std::min(outH, hStart + schedule.tileH);
      const int wStart = tw * schedule.tileW;
      const int wEnd = std::min(outW, wStart + schedule.tileW);

      for (int oh = hStart; oh < hEnd; ++oh) {
        for (int ow = wStart; ow < wEnd; ++ow) {
          float sum = 0.0f;

          if (schedule.kernelOuter) {
            for (int rr = 0; rr < ctx.workload.kernel; ++rr) {
              for (int ss = 0; ss < ctx.workload.kernel; ++ss) {
                for (int cBase = 0; cBase < ctx.workload.channels; cBase += schedule.channelBlock) {
                  const int cEnd = std::min(ctx.workload.channels, cBase + schedule.channelBlock);
                  for (int cc = cBase; cc < cEnd; ++cc) {
                    sum += ctx.input[{b,
                                      oh * ctx.workload.stride + rr,
                                      ow * ctx.workload.stride + ss,
                                      cc}] *
                           ctx.weights[{rr, ss, cc, k}];
                  }
                }
              }
            }
          } else {
            for (int cBase = 0; cBase < ctx.workload.channels; cBase += schedule.channelBlock) {
              const int cEnd = std::min(ctx.workload.channels, cBase + schedule.channelBlock);
              for (int cc = cBase; cc < cEnd; ++cc) {
                for (int rr = 0; rr < ctx.workload.kernel; ++rr) {
                  for (int ss = 0; ss < ctx.workload.kernel; ++ss) {
                    sum += ctx.input[{b,
                                      oh * ctx.workload.stride + rr,
                                      ow * ctx.workload.stride + ss,
                                      cc}] *
                           ctx.weights[{rr, ss, cc, k}];
                  }
                }
              }
            }
          }

          ctx.output[{b, oh, ow, k}] = sum;
        }
      }
    }
  };

  for (int i = 0; i < threads; ++i) {
    workers.emplace_back(worker);
  }

  for (auto& thread : workers) {
    thread.join();
  }
}

double benchmark_schedule(const ConvWorkload& workload,
                          const ConvSchedule& schedule,
                          int warmupIters,
                          int benchmarkIters) {
  ConvContext context(workload);
  fill_tensor(context.input, 0.5f, 0.1f);
  fill_tensor(context.weights, 0.25f, -0.05f);

  for (int i = 0; i < warmupIters; ++i) {
    run_conv_schedule(context, schedule);
  }

  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < benchmarkIters; ++i) {
    run_conv_schedule(context, schedule);
  }
  const auto t1 = std::chrono::high_resolution_clock::now();

  const std::chrono::duration<double, std::milli> elapsed = t1 - t0;
  return elapsed.count() / static_cast<double>(benchmarkIters);
}

bool schedules_equal(const ConvSchedule& a, const ConvSchedule& b) {
  return a.tileH == b.tileH && a.tileW == b.tileW && a.channelBlock == b.channelBlock &&
         a.numThreads == b.numThreads && a.kernelOuter == b.kernelOuter;
}

}  // namespace

std::string schedule_to_string(const ConvSchedule& schedule) {
  return "tileH=" + std::to_string(schedule.tileH) + ", tileW=" +
         std::to_string(schedule.tileW) + ", cBlock=" + std::to_string(schedule.channelBlock) +
         ", threads=" + std::to_string(schedule.numThreads) +
         ", order=" + (schedule.kernelOuter ? "kernel-outer" : "channel-outer");
}

ConvWorkload workload_from_name(const std::string& name) {
  if (name == "small") {
    return ConvWorkload{1, 28, 28, 16, 32, 3, 1, 1};
  }
  if (name == "resnet-stage") {
    return ConvWorkload{1, 56, 56, 64, 64, 3, 1, 1};
  }
  if (name == "vgg-stage") {
    return ConvWorkload{1, 112, 112, 64, 128, 3, 1, 1};
  }
  throw std::runtime_error("Unknown workload: " + name + ". Use small|resnet-stage|vgg-stage.");
}

std::vector<ConvSchedule> build_candidate_pool(int maxThreads) {
  const std::vector<int> tileHs = {1, 2, 4, 8};
  const std::vector<int> tileWs = {1, 2, 4, 8};
  const std::vector<int> channelBlocks = {1, 2, 4, 8, 16};

  std::vector<int> threads = {1, 2, 4, 8, 16};
  threads.erase(std::remove_if(threads.begin(),
                               threads.end(),
                               [maxThreads](int t) { return t > std::max(1, maxThreads); }),
                threads.end());
  if (threads.empty()) {
    threads.push_back(1);
  }

  std::vector<ConvSchedule> pool;
  for (int tileH : tileHs) {
    for (int tileW : tileWs) {
      for (int cBlock : channelBlocks) {
        for (int threadCount : threads) {
          pool.push_back(ConvSchedule{tileH, tileW, cBlock, threadCount, false});
          pool.push_back(ConvSchedule{tileH, tileW, cBlock, threadCount, true});
        }
      }
    }
  }
  return pool;
}

double evaluate_schedule(const ConvWorkload& workload,
                         const ConvSchedule& schedule,
                         int warmupIters,
                         int benchmarkIters) {
  return benchmark_schedule(workload, schedule, warmupIters, benchmarkIters);
}

SearchResult run_grid_search(const ConvWorkload& workload,
                             const SearchConfig& config,
                             const std::vector<ConvSchedule>& pool) {
  SearchResult result{"grid-search", pool.front(), std::numeric_limits<double>::infinity(), 0};

  const std::size_t maxEval =
      config.gridMaxEvals > 0 ? std::min<std::size_t>(pool.size(), static_cast<std::size_t>(config.gridMaxEvals))
                              : pool.size();

  for (std::size_t i = 0; i < maxEval; ++i) {
    const double ms = evaluate_schedule(workload, pool[i], config.warmupIters, config.benchmarkIters);
    ++result.evaluatedCount;
    if (ms < result.bestMs) {
      result.bestMs = ms;
      result.bestSchedule = pool[i];
    }
  }
  return result;
}

SearchResult run_random_search(const ConvWorkload& workload,
                               const SearchConfig& config,
                               const std::vector<ConvSchedule>& pool,
                               unsigned int seed) {
  SearchResult result{"random-search", pool.front(), std::numeric_limits<double>::infinity(), 0};

  std::mt19937 rng(seed);
  std::uniform_int_distribution<std::size_t> pick(0, pool.size() - 1);

  const int budget = std::max(1, config.randomBudget);
  for (int i = 0; i < budget; ++i) {
    const ConvSchedule& candidate = pool[pick(rng)];
    const double ms = evaluate_schedule(workload, candidate, config.warmupIters, config.benchmarkIters);
    ++result.evaluatedCount;
    if (ms < result.bestMs) {
      result.bestMs = ms;
      result.bestSchedule = candidate;
    }
  }
  return result;
}

SearchResult run_hill_climb_search(const ConvWorkload& workload,
                                   const SearchConfig& config,
                                   const std::vector<ConvSchedule>& pool,
                                   unsigned int seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<std::size_t> pick(0, pool.size() - 1);

  ConvSchedule current = pool[pick(rng)];
  double currentMs = evaluate_schedule(workload, current, config.warmupIters, config.benchmarkIters);

  SearchResult result{"hill-climb", current, currentMs, 1};

  for (int step = 0; step < std::max(1, config.hillSteps); ++step) {
    bool improved = false;

    for (const ConvSchedule& neighbor : pool) {
      int diff = 0;
      diff += current.tileH != neighbor.tileH;
      diff += current.tileW != neighbor.tileW;
      diff += current.channelBlock != neighbor.channelBlock;
      diff += current.numThreads != neighbor.numThreads;
      diff += current.kernelOuter != neighbor.kernelOuter;

      if (diff != 1) {
        continue;
      }

      const double ms = evaluate_schedule(workload, neighbor, config.warmupIters, config.benchmarkIters);
      ++result.evaluatedCount;

      if (ms < currentMs) {
        current = neighbor;
        currentMs = ms;
        improved = true;
      }
    }

    if (!improved) {
      break;
    }
  }

  result.bestSchedule = current;
  result.bestMs = currentMs;
  return result;
}
