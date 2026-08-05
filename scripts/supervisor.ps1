# Internal: runs one command forever, hidden, restarting it if it ever exits
# (crash, MT5 disconnect that bubbles past the loop's own try/except, etc).
# Not meant to be run directly -- start_all.ps1 launches one of these per process.
param(
    [Parameter(Mandatory=$true)][string]$Name,
    [Parameter(Mandatory=$true)][string]$CommandLine
)

$root = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$superLog = Join-Path $logDir "$Name`_supervisor.log"
$stdoutLog = Join-Path $logDir "$Name`_stdout.log"
$stderrLog = Join-Path $logDir "$Name`_stderr.log"

"$(Get-Date -Format o) supervisor (PID $PID) starting for '$Name': $CommandLine" | Out-File -FilePath $superLog -Append -Encoding utf8

while ($true) {
    "$(Get-Date -Format o) launching: $CommandLine" | Out-File -FilePath $superLog -Append -Encoding utf8
    $proc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $CommandLine `
        -WorkingDirectory $root -NoNewWindow -PassThru -Wait `
        -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
    "$(Get-Date -Format o) '$Name' exited with code $($proc.ExitCode) -- restarting in 5s" | Out-File -FilePath $superLog -Append -Encoding utf8
    Start-Sleep -Seconds 5
}
