$ErrorActionPreference = 'Stop'

Write-Host "[1/4] Checking Python 3.11..."
$python = Get-Command py -ErrorAction SilentlyContinue
if ($python) {
    $pyVersion = & py -3.11 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
    if ($LASTEXITCODE -ne 0 -or $pyVersion -ne "3.11") {
        throw "Python 3.11 is required. Install it first (e.g. winget install Python.Python.3.11)."
    }
    $pythonCmd = "py -3.11"
} else {
    throw "Python launcher (py) not found. Install Python 3.11 from python.org or winget."
}

Write-Host "[2/4] Creating virtual environment..."
Invoke-Expression "$pythonCmd -m venv .venv"

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment python not found at $venvPython"
}

Write-Host "[3/4] Installing project dependencies..."
& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -e .

Write-Host "[4/4] Downloading model checkpoints (first run can take a while)..."
& $venvPython tools\warmup_models.py --models all

Write-Host "Bootstrap complete."
Write-Host "Start apps with: .\start_standard.ps1, .\start_turbo.ps1, .\start_multilingual.ps1, .\start_vc.ps1"
