param(
  [string]$Workload = 'small',
  [int]$Warmup = 1,
  [int]$Iters = 2,
  [int]$RandomBudget = 40,
  [int]$HillSteps = 6,
  [int]$MaxThreads = 8,
  [int]$GridMaxEvals = 150
)

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $scriptDir
$buildDir = Join-Path $rootDir 'build'

cmake -S $rootDir -B $buildDir
cmake --build $buildDir --config Release

$exePath = Join-Path $buildDir 'Release/ttl_schedule_search_demo.exe'
if (-not (Test-Path $exePath)) {
  $exePath = Join-Path $buildDir 'ttl_schedule_search_demo.exe'
}

$args = @(
  '--workload', $Workload,
  '--warmup', $Warmup,
  '--iters', $Iters,
  '--random-budget', $RandomBudget,
  '--hill-steps', $HillSteps,
  '--max-threads', $MaxThreads,
  '--grid-max-evals', $GridMaxEvals
)

& $exePath $args
