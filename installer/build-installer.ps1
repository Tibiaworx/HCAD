# Build the HCAD release exe and package the Windows installer (HCAD-Setup.exe).
# Usage:  powershell -ExecutionPolicy Bypass -File installer\build-installer.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent

# 1) Release build (CMake on PATH for the Manifold C++ dependency).
Push-Location $root
$env:PATH = "C:\Program Files\CMake\bin;$env:PATH"
cargo build --release -p hworks-app
Pop-Location

# 2) The installer bundles vc_redist.x64.exe - fetch it once if missing.
$redist = Join-Path $PSScriptRoot "vc_redist.x64.exe"
if (-not (Test-Path $redist)) {
    Write-Host "Downloading vc_redist.x64.exe..."
    Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile $redist -UseBasicParsing
}

# 3) Package with NSIS.
$makensis = "C:\Program Files (x86)\NSIS\makensis.exe"
if (-not (Test-Path $makensis)) { throw "NSIS not found - install with: winget install -e --id NSIS.NSIS" }
& $makensis (Join-Path $PSScriptRoot "HCAD.nsi")

Write-Host "`nInstaller ready: $(Join-Path $PSScriptRoot 'HCAD-Setup.exe')"
