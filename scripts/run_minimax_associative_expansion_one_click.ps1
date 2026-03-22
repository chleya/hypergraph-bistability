Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$env:ANTHROPIC_API_KEY = "sk-cp-32X4yB3hv4uMfzdBmke7EyaxE2pXmHkAGisoBxm1bTlSnUKXcH3lGRgWYcD62Nre5AacJpbi0E5yOx92m5rkIth9HioW2aCHP5r3LeCKBuf-wdr1TVgeFxY"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

python -m hypergraph_bistability.cli run-mechanism-experiment `
  --experiment associative_expansion `
  --base-url https://api.minimaxi.com/anthropic `
  --model MiniMax-M2.7 `
  --force-powershell `
  --output _experiment_associative_expansion_llm.json

Write-Host ""
Write-Host "Result file: $projectRoot\\_experiment_associative_expansion_llm.json"
