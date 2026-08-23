$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = if ($env:PYTHON) { $env:PYTHON } else { (Get-Command python).Source }
$Data = Join-Path $Root "data_panel_history_main_chinext_20220101_20260626.csv"
$Benchmark = "D:\codex\量化\data\processed\510300.csv"
$Output = Join-Path $Root "outputs\high_return_v2\next_open_rank_model_stock_focus_expanded"

& $Python (Join-Path $Root "train_next_open_rank_model.py") `
  --data $Data `
  --output-dir $Output `
  --rebalance-frequency 5 `
  --top-n 40 `
  --max-position-weight 0.029 `
  --leverage 0.93 `
  --benchmark $Benchmark `
  --market-ma-window 120 `
  --market-below-ma-exposure 0.65 `
  --market-risk-off-drawdown-20d -0.08 `
  --market-crash-exposure 0.10
