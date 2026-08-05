# Quick check: which of the 3 background processes are alive right now, plus
# a sweep for orphan duplicates that no pid file is tracking (see
# process_utils.ps1 for why that's a real, seen-in-production failure mode).
$root = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $root "run"

. (Join-Path $PSScriptRoot "process_utils.ps1")

$jobs = @(
    @{ Name = "realtime_loop"; Module = "src.realtime_loop" },
    @{ Name = "intraday_loop"; Module = "src.intraday_loop" },
    @{ Name = "widget";        Module = "src.widget" }
)

foreach ($j in $jobs) {
    $pidFile = Join-Path $runDir "$($j.Name).supervisor.pid"
    $trackedPid = $null
    if (Test-Path $pidFile) {
        $filePid = Get-Content $pidFile -ErrorAction SilentlyContinue
        if ($filePid -and (Test-IsTrackedSupervisor -ProcId $filePid -JobName $j.Name)) {
            $trackedPid = [int]$filePid
        }
    }

    if ($trackedPid) {
        Write-Host "$($j.Name) : RUNNING (supervisor PID $trackedPid)"
    } elseif (Test-Path $pidFile) {
        Write-Host "$($j.Name) : STOPPED (stale pid file)"
    } else {
        Write-Host "$($j.Name) : NOT STARTED"
    }

    $authorized = @()
    if ($trackedPid) { $authorized = @($trackedPid) + (Get-DescendantProcessIds -RootProcId $trackedPid) }
    $orphans = Get-OrphanModuleProcesses -Module $j.Module -AuthorizedPids $authorized
    foreach ($o in $orphans) {
        Write-Host "    WARNING: untracked duplicate PID $($o.ProcessId) also running ($($o.CommandLine.Trim()))"
    }
}
