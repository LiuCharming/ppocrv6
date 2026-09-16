param([int]$Port = 8001, [int]$Concurrency = 2, [int]$QueueSize = 16)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv-cpu\Scripts\python.exe'
if (!(Test-Path -LiteralPath $pythonPath)) {
    throw 'Create .venv-cpu with Python 3.11 and install requirements-cpu.txt first. See README.md.'
}
$env:OCR_DEVICE = 'cpu'
$env:OCR_CONCURRENCY = $Concurrency
$env:OCR_QUEUE_SIZE = $QueueSize
$env:PADDLE_PDX_MODEL_SOURCE = 'BOS'
& $pythonPath -m uvicorn model_service.main:app --host 127.0.0.1 --port $Port --workers 1
