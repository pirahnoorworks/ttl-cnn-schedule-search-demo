#include <algorithm>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "schedule_search.h"

namespace {

struct CliConfig {
  std::string workload = "small";
  int warmup = 1;
  int iters = 2;
  int randomBudget = 40;
  int hillSteps = 6;
  int maxThreads = 8;
  int gridMaxEvals = 150;
};

bool readInt(const std::vector<std::string>& args,
             size_t& index,
             const std::string& flag,
             int& out) {
  if (args[index] != flag) {
    return false;
  }
  if (index + 1 >= args.size()) {
    throw std::runtime_error("Missing value for " + flag);
  }
  out = std::stoi(args[++index]);
  return true;
}

CliConfig parseArgs(int argc, char** argv) {
  CliConfig cfg;
  std::vector<std::string> args(argv + 1, argv + argc);

  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--workload") {
      if (i + 1 >= args.size()) {
        throw std::runtime_error("Missing value for --workload.");
      }
      cfg.workload = args[++i];
      continue;
    }
    if (readInt(args, i, "--warmup", cfg.warmup)) {
      continue;
    }
    if (readInt(args, i, "--iters", cfg.iters)) {
      continue;
    }
    if (readInt(args, i, "--random-budget", cfg.randomBudget)) {
      continue;
    }
    if (readInt(args, i, "--hill-steps", cfg.hillSteps)) {
      continue;
    }
    if (readInt(args, i, "--max-threads", cfg.maxThreads)) {
      continue;
    }
    if (readInt(args, i, "--grid-max-evals", cfg.gridMaxEvals)) {
      continue;
    }
    if (args[i] == "--help" || args[i] == "-h") {
      std::cout
          << "Usage: ttl_schedule_search_demo [options]\n"
          << "  --workload <name>       small|resnet-stage|vgg-stage (default small)\n"
          << "  --warmup <int>          Warmup iterations per candidate (default 1)\n"
          << "  --iters <int>           Timed iterations per candidate (default 2)\n"
          << "  --random-budget <int>   Number of random candidates (default 40)\n"
          << "  --hill-steps <int>      Max hill-climb improvement rounds (default 6)\n"
          << "  --max-threads <int>     Max thread count in candidate pool (default 8)\n"
          << "  --grid-max-evals <int>  Max candidates for grid search (default 150)\n";
      std::exit(0);
    }
    throw std::runtime_error("Unknown argument: " + args[i]);
  }

  return cfg;
}

void validate(const CliConfig& cfg) {
  if (cfg.warmup < 0 || cfg.iters <= 0 || cfg.randomBudget <= 0 || cfg.hillSteps <= 0 ||
      cfg.maxThreads <= 0 || cfg.gridMaxEvals <= 0) {
    throw std::runtime_error("Iteration and budget values must be positive (warmup >= 0).");
  }
}

void print_workload(const ConvWorkload& w) {
  std::cout << "Workload: N=" << w.batch << ", H=" << w.height << ", W=" << w.width
            << ", C=" << w.channels << ", K=" << w.filters << ", R=S=" << w.kernel
            << ", stride=" << w.stride << ", pad=" << w.pad << "\n";
}

void print_result(const SearchResult& result, double baselineMs) {
  std::cout << std::fixed << std::setprecision(4);
  std::cout << result.technique << " | best_ms=" << result.bestMs
            << " | speedup_vs_baseline=" << (baselineMs / result.bestMs)
            << " | evals=" << result.evaluatedCount << "\n";
  std::cout << "  best_schedule: " << schedule_to_string(result.bestSchedule) << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliConfig cli = parseArgs(argc, argv);
    validate(cli);

    ConvWorkload workload = workload_from_name(cli.workload);
    const int hardwareThreads = static_cast<int>(std::thread::hardware_concurrency());
    const int maxThreads = hardwareThreads > 0 ? std::min(cli.maxThreads, hardwareThreads) : cli.maxThreads;

    SearchConfig searchConfig{cli.warmup, cli.iters, cli.randomBudget, cli.hillSteps, maxThreads, cli.gridMaxEvals};

    auto pool = build_candidate_pool(searchConfig.maxThreads);
    ConvSchedule baseline{1, 1, 1, 1, false};
    const double baselineMs = evaluate_schedule(workload, baseline, cli.warmup, cli.iters);

    std::cout << "Tensor Template Language CNN Schedule Search Demo\n";
    print_workload(workload);
    std::cout << "Candidate pool size: " << pool.size() << " (maxThreads=" << searchConfig.maxThreads << ")\n";
    std::cout << "Baseline schedule: " << schedule_to_string(baseline) << "\n";
    std::cout << "Baseline ms: " << std::fixed << std::setprecision(4) << baselineMs << "\n\n";

    SearchResult grid = run_grid_search(workload, searchConfig, pool);
    SearchResult random = run_random_search(workload, searchConfig, pool, 1337u);
    SearchResult hill = run_hill_climb_search(workload, searchConfig, pool, 2026u);

    print_result(grid, baselineMs);
    print_result(random, baselineMs);
    print_result(hill, baselineMs);

    std::vector<SearchResult> all = {grid, random, hill};
    auto best = std::min_element(all.begin(), all.end(), [](const SearchResult& a, const SearchResult& b) {
      return a.bestMs < b.bestMs;
    });

    std::cout << "\nOverall winner: " << best->technique << " with " << best->bestMs << " ms\n";
    std::cout << "Winning schedule: " << schedule_to_string(best->bestSchedule) << "\n";

    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}
