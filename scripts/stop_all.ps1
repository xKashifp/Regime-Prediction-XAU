# Stops all supervisors started by start_all.ps1, plus whatever python
# process each one is currently running underneath it -- and, since pid files
# only ever track one supervisor per job, also sweeps for any untracked
# "python -m src.<module>" orphan (e.g. a manual/debug supervisor.ps1 run
# under a different -Name) so this is a true full stop, not just "stop
# whatever the pid files happen to know about."
$root = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $root "run"

. (Join-Path $PSScriptRoot "process_utils.ps1")

$modules = @("src.realtime_loop", "src.intraday_loop", "src.widget")

if (Test-Path $runDir) {
    Get-ChildItem -Path $runDir -Filter "*.supervisor.pid" -ErrorAction SilentlyContinue | ForEach-Object {
        $name = $_.Name -replace '\.supervisor\.pid$', ''
        $procId = Get-Content $_.FullName -ErrorAction SilentlyContinue
        if ($procId -and (Get-Process -Id $procId -ErrorAction SilentlyContinue)) {
            Write-Host "Stopping $name (PID $procId, and its child process tree)..."
            taskkill /PID $procId /T /F | Out-Null
        } else {
            Write-Host "${name}: not running."
        }
        Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
    }
} else {
    Write-Host "No run\ directory -- nothing tracked, but still sweeping for orphans."
}

foreach ($m in $modules) {
    $orphans = Get-OrphanModuleProcesses -Module $m -AuthorizedPids @()
    foreach ($o in $orphans) {
        Write-Host "Stopping untracked orphan PID $($o.ProcessId) ($($o.CommandLine.Trim()))..."
        try { Stop-Process -Id $o.ProcessId -Force -ErrorAction Stop } catch {}
    }
}

Write-Host "Done."
