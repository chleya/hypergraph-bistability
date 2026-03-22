param(
    [Parameter(Mandatory = $true)]
    [string]$ApiKey,

    [Parameter(Mandatory = $true)]
    [string]$BaseUrl,

    [Parameter(Mandatory = $true)]
    [string]$PayloadFile,

    [Parameter(Mandatory = $true)]
    [string]$OutputFile,

    [Parameter(Mandatory = $true)]
    [ValidateSet("anthropic", "openai")]
    [string]$Transport
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
[Console]::InputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$bodyObject = Get-Content -Raw -Encoding UTF8 $PayloadFile | ConvertFrom-Json
$body = $bodyObject | ConvertTo-Json -Depth 20 -Compress

if ($Transport -eq "anthropic") {
    $headers = @{
        "x-api-key" = $ApiKey
        "anthropic-version" = "2023-06-01"
    }
    $bodyBytes = [System.Text.Encoding]::UTF8.GetBytes($body)
    $uri = ($BaseUrl.TrimEnd("/") + "/v1/messages")
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
    exit 0
}

    $headers = @{
        "Authorization" = "Bearer $ApiKey"
    }
    $bodyBytes = [System.Text.Encoding]::UTF8.GetBytes($body)
    $uri = ($BaseUrl.TrimEnd("/") + "/chat/completions")
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
