# Activate virtual environment automatically for PowerShell
if (Test-Path -Path ".venv/Scripts/Activate.ps1") {
    & ".venv/Scripts/Activate.ps1"
}
