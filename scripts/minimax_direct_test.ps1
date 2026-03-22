param(
    [Parameter(Mandatory = $false)]
    [string]$ApiKey = $env:ANTHROPIC_API_KEY,

    [Parameter(Mandatory = $false)]
    [string]$Model = "MiniMax-M2.7",

    [Parameter(Mandatory = $false)]
    [string]$Prompt = "Reply with exactly OK",

    [Parameter(Mandatory = $false)]
    [ValidateSet("anthropic", "openai", "both")]
    [string]$Transport = "anthropic",

    [Parameter(Mandatory = $false)]
    [double]$Temperature = 0.01
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $ApiKey) {
    throw "No API key provided. Pass -ApiKey or set ANTHROPIC_API_KEY / OPENAI_API_KEY."
}

function Invoke-MiniMaxAnthropic {
    param(
        [string]$Key,
        [string]$ModelName,
        [string]$UserPrompt,
        [double]$Temp
    )

    $headers = @{
        "x-api-key" = $Key
        "anthropic-version" = "2023-06-01"
        "Content-Type" = "application/json"
    }
    $body = @{
        model = $ModelName
        messages = @(
            @{
                role = "user"
                content = @(
                    @{
                        type = "text"
                        text = $UserPrompt
                    }
                )
            }
        )
        temperature = $Temp
        max_tokens = 200
    } | ConvertTo-Json -Depth 8

    $response = Invoke-WebRequest -Uri "https://api.minimaxi.com/anthropic/v1/messages" -Headers $headers -Method POST -Body $body -UseBasicParsing
    $json = $response.Content | ConvertFrom-Json
    $texts = @($json.content | Where-Object { $_.type -eq "text" } | Select-Object -ExpandProperty text)

    [pscustomobject]@{
        transport = "anthropic"
        status_code = $response.StatusCode
        raw_content = $response.Content
        text = ($texts -join "`n").Trim()
    }
}

function Invoke-MiniMaxOpenAI {
    param(
        [string]$Key,
        [string]$ModelName,
        [string]$UserPrompt,
        [double]$Temp
    )

    $headers = @{
        "Authorization" = "Bearer $Key"
        "Content-Type" = "application/json"
    }
    $body = @{
        model = $ModelName
        messages = @(
            @{
                role = "user"
                content = $UserPrompt
            }
        )
        temperature = $Temp
        max_tokens = 200
    } | ConvertTo-Json -Depth 6

    $response = Invoke-WebRequest -Uri "https://api.minimaxi.com/v1/chat/completions" -Headers $headers -Method POST -Body $body -UseBasicParsing
    $json = $response.Content | ConvertFrom-Json

    [pscustomobject]@{
        transport = "openai"
        status_code = $response.StatusCode
        raw_content = $response.Content
        text = [string]$json.choices[0].message.content
    }
}

function Show-Result {
    param([pscustomobject]$Result)

    Write-Host ""
    Write-Host "Transport: $($Result.transport)"
    Write-Host "Status: $($Result.status_code)"
    Write-Host "Text:"
    Write-Host $Result.text
    Write-Host ""
    Write-Host "Raw JSON:"
    Write-Host $Result.raw_content
}

if ($Transport -eq "anthropic" -or $Transport -eq "both") {
    try {
        $anthropicResult = Invoke-MiniMaxAnthropic -Key $ApiKey -ModelName $Model -UserPrompt $Prompt -Temp $Temperature
        Show-Result -Result $anthropicResult
    }
    catch {
        Write-Host ""
        Write-Host "Transport: anthropic"
        Write-Host "Error:"
        Write-Host $_.Exception.Message
    }
}

if ($Transport -eq "openai" -or $Transport -eq "both") {
    try {
        $openaiResult = Invoke-MiniMaxOpenAI -Key $ApiKey -ModelName $Model -UserPrompt $Prompt -Temp $Temperature
        Show-Result -Result $openaiResult
    }
    catch {
        Write-Host ""
        Write-Host "Transport: openai"
        Write-Host "Error:"
        Write-Host $_.Exception.Message
    }
}
