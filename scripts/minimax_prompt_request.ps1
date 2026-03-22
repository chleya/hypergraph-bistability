param(
    [Parameter(Mandatory = $true)]
    [string]$ApiKey,

    [Parameter(Mandatory = $true)]
    [string]$Model,

    [Parameter(Mandatory = $true)]
    [string]$PromptFile,

    [Parameter(Mandatory = $true)]
    [string]$OutputFile,

    [Parameter(Mandatory = $false)]
    [double]$Temperature = 0.01,

    [Parameter(Mandatory = $false)]
    [int]$MaxTokens = 900
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
[Console]::InputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$promptText = [string](Get-Content -Raw -Encoding UTF8 $PromptFile)

$headers = @{
    "x-api-key" = $ApiKey
    "anthropic-version" = "2023-06-01"
}
$body = @{
    model = $Model
    messages = @(
        @{
            role = "user"
            content = @(
                @{
                    type = "text"
                    text = $promptText
                }
            )
        }
    )
    temperature = $Temperature
    max_tokens = $MaxTokens
} | ConvertTo-Json -Depth 8

$bodyBytes = [System.Text.Encoding]::UTF8.GetBytes($body)
$uri = "https://api.minimaxi.com/anthropic/v1/messages"
$request = [System.Net.HttpWebRequest]::Create($uri)
$request.Method = "POST"
$request.ContentType = "application/json; charset=utf-8"
foreach ($key in $headers.Keys) {
    $request.Headers.Add($key, $headers[$key])
}
$request.ContentLength = $bodyBytes.Length
$stream = $request.GetRequestStream()
$stream.Write($bodyBytes, 0, $bodyBytes.Length)
$stream.Close()
$response = $request.GetResponse()
$responseStream = $response.GetResponseStream()
$reader = New-Object System.IO.StreamReader($responseStream, [System.Text.Encoding]::UTF8, $true)
$responseText = $reader.ReadToEnd()
$reader.Dispose()
Set-Content -Path $OutputFile -Value $responseText -Encoding UTF8
