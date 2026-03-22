$ErrorActionPreference = "Stop"

$output = "F:\hypergraph_bistability\_experiment_procedure_memory_llm.json"

if (-not $env:ANTHROPIC_API_KEY) {
  $env:ANTHROPIC_API_KEY = "sk-cp-32X4yB3hv4uMfzdBmke7EyaxE2pXmHkAGisoBxm1bTlSnUKXcH3lGRgWYcD62Nre5AacJpbi0E5yOx92m5rkIth9HioW2aCHP5r3LeCKBuf-wdr1TVgeFxY"
}

python -m hypergraph_bistability.cli run-mechanism-experiment `
  --experiment procedure_memory `
  --base-url https://api.minimaxi.com/anthropic `
  --model MiniMax-M2.7 `
  --force-powershell `
  --output $output

Write-Host ""
Write-Host "Result file: $output"
