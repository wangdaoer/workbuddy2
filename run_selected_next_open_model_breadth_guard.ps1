$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = if ($env:PYTHON) { $env:PYTHON } else { (Get-Command python).Source }
$Data = Join-Path $Root "data_panel_history_main_chinext_20220101_20260626.csv"
$Benchmark = "D:\codex\量化\data\processed\510300.csv"
$Output = Join-Path $Root "outputs\high_return_v2\next_open_rank_model_selected_breadth_guard"

& $Python (Join-Path $Root "train_next_open_rank_model.py") `
  --data $Data `
  --output-dir $Output `
  --rebalance-frequency 5 `
  --top-n 40 `
  --max-position-weight 0.028 `
  --leverage 0.90 `
  --benchmark $Benchmark `
  --market-ma-window 120 `
  --market-below-ma-exposure 0.60 `
  --market-risk-off-drawdown-20d -0.08 `
  --market-crash-exposure 0.0 `
  --breadth-filter `
  --breadth-ma-window 60 `
  --breadth-threshold 0.45 `
  --breadth-below-exposure 0.55 `
  --breadth-crash-threshold 0.32 `
  --breadth-crash-exposure 0.20
