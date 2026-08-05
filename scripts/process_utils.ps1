# Shared process-tracking helpers for start_all.ps1 / status_all.ps1 / stop_all.ps1.
#
# The pid files under run\ only ever record ONE pid per job: the top-level
# supervisor.ps1 process. That's fine as long as nothing else ever launches a
# second "python -m src.<module>" outside that tracking -- but a manual/debug
# supervisor.ps1 invocation under a different -Name (or a pid file overwritten
# by a second start while the first was still alive) produces exactly that: a
# second writer to the same DB that no pid file points to and no status check
# ever surfaces. These helpers find those by actual command line, not by trusting
# whichever single pid happens to be on file.

function Get-DescendantProcessIds {
    param([Parameter(Mandatory=$true)][int]$RootProcId)
    $result = @()
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$RootProcId" -ErrorAction SilentlyContinue
    foreach ($c in $children) {
        $result += [int]$c.ProcessId
        $result += Get-DescendantProcessIds -RootProcId $c.ProcessId
    }
    return $result
}

function Test-IsTrackedSupervisor {
    # A pid file entry is only trustworthy if the pid still refers to *our*
    # supervisor.ps1 for *this* job -- not just any process (Windows recycles
    # pid numbers, so Get-Process -Id existing is not proof by itself).
    param([Parameter(Mandatory=$true)]$ProcId, [Parameter(Mandatory=$true)][string]$JobName)
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcId" -ErrorAction SilentlyContinue
    if (-not $proc) { return $false }
    return ($proc.CommandLine -match "supervisor\.ps1") -and ($proc.CommandLine -match [regex]::Escape($JobName))
}

function Get-OrphanModuleProcesses {
    # Any "python -m <module>" process whose pid isn't in $AuthorizedPids --
    # i.e. not a descendant of the one supervisor we trust for this job.
    param([Parameter(Mandatory=$true)][string]$Module, [int[]]$AuthorizedPids = @())
    $pattern = "-m\s+" + [regex]::Escape($Module) + "(\s|$)"
    $matches_ = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match $pattern }
    return $matches_ | Where-Object { $AuthorizedPids -notcontains [int]$_.ProcessId }
}
