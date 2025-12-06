$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
while ($true) {
    if (Test-Path $python) {
        & $python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            [console]::beep(880,400)
            [console]::beep(988,400)
            [console]::beep(1046,400)
            break
        }
    }
    Start-Sleep -Seconds 10
}
