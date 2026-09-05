# --- 1. Vérifier que Python 3.12 est disponible -------------------------
$py312 = (py -0p 2>&1 | Select-String "3\.12").Line
if (-not $py312) {
    Write-Host "Python 3.12 introuvable. Installez-le depuis python.org, puis relancez." -ForegroundColor Red
    exit 1
}

# --- 2. Recréer le venv ------------------------------------------------
if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
py -3.12 -m venv .venv
& .\.venv\Scripts\Activate.ps1

$ver = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($ver -ne "3.12") {
    Write-Host "Le venv rapporte Python $ver, pas 3.12. Arret." -ForegroundColor Red
    exit 1
}
Write-Host "venv en Python $ver" -ForegroundColor Green

# --- 3. Dependances Python --------------------------------------------
python -m pip install --upgrade pip
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "L'installation a echoue. Collez la sortie ci-dessus." -ForegroundColor Red
    exit 1
}

# --- 4. Verifier ce qui manquait --------------------------------------
python -c "import openai, pandas, sentence_transformers, pypdf; print('OK: openai, pandas, sentence-transformers, pypdf')"

# --- 5. Frontend -------------------------------------------------------
Push-Location frontend
npm install
Pop-Location

Write-Host ""
Write-Host "Termine. Deux terminaux a ouvrir :" -ForegroundColor Green
Write-Host "  1) .venv\Scripts\Activate.ps1 ; `$env:ALLOW_UNAUTHENTICATED='1' ; uvicorn api.server:app --reload --port 8000"
Write-Host "  2) cd frontend ; npm run dev"