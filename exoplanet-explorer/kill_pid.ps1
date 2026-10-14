$ErrorActionPreference = "Stop"
$pid = 13092
$p = Get-Process -Id $pid -ErrorAction SilentlyContinue
if ($p) {
    Write-Host "Killing PID $pid"
    $p.Kill()
    $p.WaitForExit(5000)
}
Start-Sleep -Seconds 4
$uri = "http://localhost:8501"
$client = New-Object System.Net.WebClient
try {
    $html = $client.DownloadString($uri)
    Write-Host "Fetch OK, bytes: $($html.Length)"
} catch {
    Write-Host "Fetch failed: $_"
}
