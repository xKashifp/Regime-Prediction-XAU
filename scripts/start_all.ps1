# Starts realtime_loop, intraday_loop, and the widget in the background --
# no terminal window needs to stay open. Each runs under its own supervisor
# that auto-restarts it if it ever crashes. Safe to re-run: already-running
# processes are left alone, not duplicated.
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\start_all.ps1
# Stop:   powershell -ExecutionPolicy Bypass -File scripts\stop_all.ps1
# Status: powershell -ExecutionPolicy Bypass -File scripts\status_all.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $root "run"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

. (Join-Path $PSScriptRoot "process_utils.ps1")

$jobs = @(
    @{ Name = "realtime_loop"; Cmd = "python -m src.realtime_loop"; Module = "src.realtime_loop" },
    @{ Name = "intraday_loop"; Cmd = "python -m src.intraday_loop"; Module = "src.intraday_loop" },
    @{ Name = "widget";        Cmd = "python -m src.widget";        Module = "src.widget" }
)

foreach ($j in $jobs) {
    $pidFile = Join-Path $runDir "$($j.Name).supervisor.pid"
    $trackedPid = $null
    if (Test-Path $pidFile) {
        $oldPid = Get-Content $pidFile -ErrorAction SilentlyContinue
        # Get-Process only proves *some* process holds that PID number -- Windows
        # recycles PIDs, so also check it's actually our own supervisor.ps1 for
        # this job, not an unrelated process that happens to reuse the number.
        if ($oldPid -and (Test-IsTrackedSupervisor -ProcId $oldPid -JobName $j.Name)) {
            $trackedPid = [int]$oldPid
        }
    }

    # Orphan sweep: any "python -m <module>" process not descended from the
    # tracked supervisor (e.g. a manual/debug supervisor.ps1 invocation under a
    # different -Name, or a supervisor whose pid file got overwritten by a
    # second start while the first was still alive) is a silent duplicate
    # writer to the same DB. Kill it before deciding whether to start fresh.
    $authorized = @()
    if ($trackedPid) { $authorized = @($trackedPid) + (Get-DescendantProcessIds -RootProcId $trackedPid) }
    $orphans = Get-OrphanModuleProcesses -Module $j.Module -AuthorizedPids $authorized
    foreach ($o in $orphans) {
        Write-Host "$($j.Name): found orphan PID $($o.ProcessId) ($($o.CommandLine.Trim())) -- killing."
        try { Stop-Process -Id $o.ProcessId -Force -ErrorAction Stop } catch {}
    }

    if ($trackedPid) {
        Write-Host "$($j.Name): already running (supervisor PID $trackedPid) -- skipping."
        continue
    }

    # Every value that can contain a space (the script path itself -- this
    # folder is "Regime Prediction" -- and the job command line) must be
    # individually wrapped in embedded double quotes here. Start-Process's
    # array -ArgumentList does NOT reliably re-quote elements containing
    # spaces on Windows PowerShell 5.1: an unquoted path/command silently
    # gets split at the space, "-File" only sees half the path, and the
    # whole supervisor.ps1 launch fails immediately with no log written
    # anywhere (confirmed via diagnostic -- this was a real bug, not a
    # one-off environment issue).
    $scriptPathQuoted = '"' + (Join-Path $PSScriptRoot "supervisor.ps1") + '"'
    $cmdQuoted = '"' + $j.Cmd + '"'
    $p = Start-Process -FilePath "powershell.exe" -WindowStyle Hidden -PassThru -ArgumentList @(
        "-NoProfile", "-WindowStyle", "Hidden", "-ExecutionPolicy", "Bypass",
        "-File", $scriptPathQuoted,
        "-Name", $j.Name, "-CommandLine", $cmdQuoted
    )
    Set-Content -Path $pidFile -Value $p.Id -Encoding ascii
    Write-Host "$($j.Name): started (supervisor PID $($p.Id))."
}

Write-Host ""
Write-Host "Logs:   $root\logs\<name>_stdout.log / _stderr.log / _supervisor.log"
Write-Host "Stop:   powershell -ExecutionPolicy Bypass -File scripts\stop_all.ps1"
Write-Host "Status: powershell -ExecutionPolicy Bypass -File scripts\status_all.ps1"
