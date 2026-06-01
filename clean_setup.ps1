$ErrorActionPreference = "Stop"
$Venv = "tsunagu-venv"

Write-Host "[1/4] Cleaning previous environment..."
foreach ($d in @($Venv, ".pytest_cache", ".mypy_cache", ".ruff_cache")) {
    if (Test-Path $d) { Remove-Item -Recurse -Force $d }
}

Write-Host "[2/4] Creating virtual environment: $Venv (Python 3.12)"
py -3.12 -m venv $Venv

Write-Host "[3/4] Installing CPU torch, then the project with dev dependencies..."
& "$Venv\Scripts\python.exe" -m pip install --upgrade pip
& "$Venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
& "$Venv\Scripts\python.exe" -m pip install -e ".[dev]"

Write-Host "[4/4] Installing pre-commit hooks..."
& "$Venv\Scripts\pre-commit.exe" install

Write-Host "Setup complete. Activate with: .\$Venv\Scripts\Activate.ps1"
