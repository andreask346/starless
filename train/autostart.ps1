# Starless training autostart — relaunches the queue after a reboot.
# Registered as a logon scheduled task. Idempotent & guarded:
#   - does nothing if the queue is already running (no double-launch / GPU clash)
#   - does nothing if every queued experiment is already DONE
#   - does nothing if a STOP file is present (kill switch respected)
# Otherwise it resumes the queue from the on-disk checkpoints.

$root = "C:\Users\User\Documents\For Claude\starless"
Set-Location $root
$logline = "{0}  autostart fired" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')

# guard 1: already running?
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*run_queue.py*" }
if ($running) { "$logline -> already running, skip" | Add-Content "$root\runs\autostart.log"; exit 0 }

# guard 2: kill switch
if (Test-Path "$root\STOP") { "$logline -> STOP present, skip" | Add-Content "$root\runs\autostart.log"; exit 0 }

# guard 3: everything already done?
try {
    $txt = (Get-Content "$root\train\queue.json" -Raw) -replace '^\xEF\xBB\xBF', ''
    $queue = $txt | ConvertFrom-Json
    $allDone = $true
    foreach ($e in $queue) { if (-not (Test-Path "$root\runs\$($e.name)\DONE")) { $allDone = $false } }
    if ($allDone) { "$logline -> all experiments DONE, skip" | Add-Content "$root\runs\autostart.log"; exit 0 }
} catch { }

# launch (resumes from checkpoints; skips DONE experiments internally)
$env:PYTHONIOENCODING = "utf-8"
Start-Process -WindowStyle Hidden -FilePath "$root\.venv\Scripts\python.exe" `
    -ArgumentList "train\run_queue.py" `
    -RedirectStandardOutput "$root\runs\queue.out" `
    -RedirectStandardError "$root\runs\queue.err2" `
    -WorkingDirectory $root
"$logline -> LAUNCHED queue" | Add-Content "$root\runs\autostart.log"
exit 0
