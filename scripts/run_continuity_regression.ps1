$ErrorActionPreference = "Stop"

$projectRoot = "F:\hypergraph_bistability"
Set-Location $projectRoot

$output = Join-Path $projectRoot "_continuity_regression.json"

python -m hypergraph_bistability.cli run-continuity-regression --output $output

Write-Host ""
Write-Host "Result file: $output"
