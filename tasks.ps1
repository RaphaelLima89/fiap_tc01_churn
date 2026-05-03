<#
.SYNOPSIS
    Equivalente PowerShell ao Makefile, pra desenvolvimento no Windows sem GNU make.
.EXAMPLE
    .\tasks.ps1 test
    .\tasks.ps1 lint
#>

param(
    [Parameter(Position = 0)]
    [ValidateSet("install", "lint", "format", "test", "run", "train", "clean", "help")]
    [string]$Task = "help"
)

switch ($Task) {
    "install" { poetry install }
    "lint"    { poetry run ruff check src tests }
    "format"  { poetry run ruff format src tests }
    "test"    { poetry run pytest }
    "run"     { poetry run uvicorn churn_predictor.api.main:app --reload --port 8000 }
    "train"   {
        poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
        poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/02_modelos.ipynb
    }
    "clean"   {
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue .pytest_cache, .ruff_cache
        Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
    }
    "help"    {
        Write-Host "Tasks disponiveis:"
        Write-Host "  install   - poetry install"
        Write-Host "  lint      - ruff check src tests"
        Write-Host "  format    - ruff format src tests"
        Write-Host "  test      - pytest"
        Write-Host "  run       - uvicorn porta 8000"
        Write-Host "  train     - executa notebooks 01 e 02"
        Write-Host "  clean     - remove caches"
    }
}