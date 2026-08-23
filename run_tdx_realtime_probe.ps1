param(
    [int]$Samples = 8,
    [double]$IntervalSeconds = 30,
    [string]$Codes = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$ArgsList = @(
    "tdx_realtime_probe.py",
    "--samples", "$Samples",
    "--interval-seconds", "$IntervalSeconds"
)

if ($Codes.Trim() -ne "") {
    $ArgsList += @("--codes", $Codes)
}

python @ArgsList
