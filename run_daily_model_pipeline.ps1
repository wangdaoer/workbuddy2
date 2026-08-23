$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = if ($env:PYTHON) { $env:PYTHON } else { (Get-Command python).Source }

& $Python (Join-Path $Root "run_daily_model_pipeline.py") @args
