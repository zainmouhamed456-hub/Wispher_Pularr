$ErrorActionPreference = "Stop"

Write-Host "Omnilingual ASR cannot be installed reliably in native Windows Python." -ForegroundColor Yellow
Write-Host "The blocker is fairseq2n: Meta publishes the compatible wheel for Linux, not win_amd64."
Write-Host ""
Write-Host "Use WSL/Linux, then run:"
Write-Host "  bash scripts/setup_omnilingual_eval.sh"
Write-Host ""
Write-Host "If WSL has no distro installed yet:"
Write-Host "  wsl --list --online"
Write-Host "  wsl --install Ubuntu-24.04"
Write-Host ""
Write-Host "If Docker is available on a Linux-capable machine:"
Write-Host "  docker build -f Dockerfile.omnilingual -t whisper-pularr-omni ."
Write-Host "  docker run --rm -it -v `"`$PWD:/workspace`" whisper-pularr-omni"
Write-Host ""
exit 1
