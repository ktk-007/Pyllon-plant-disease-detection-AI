# PlantAI Automated Training and Export Pipeline (RESUME-AWARE)
# Optimized for RTX 4050 — skips already finished models.$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"

$Plants = @("tomato", "mango_leaf", "mango_fruit", "apple", "corn", "grape", "potato", "bellpepper", "strawberry", "rose")
$Models = @("convnext", "effnet")

foreach ($Plant in $Plants) {
    foreach ($Model in $Models) {
        $QuantizedPath = "models/${Model}_${Plant}.pth"
        
        if (Test-Path $QuantizedPath) {
            Write-Host "SKIPPING: $Plant | $Model (Already finished)" -ForegroundColor Yellow
            continue
        }

        Write-Host "==========================================================" -ForegroundColor Cyan
        Write-Host "STARTING: $Plant | $Model" -ForegroundColor Green
        Write-Host "==========================================================" -ForegroundColor Cyan
        
        # 1. Train
        .\venv\Scripts\python.exe train.py --plant $Plant --model $Model --batch_size 32
        
        # Check if train succeeded
        if ($LASTEXITCODE -eq 0) {
            # 2. Export
            .\venv\Scripts\python.exe export.py --plant $Plant --model $Model
            Write-Host "COMPLETED: $Plant | $Model" -ForegroundColor Green
        } else {
            Write-Host "ERROR: Training failed for $Plant $Model. Moving to next." -ForegroundColor Red
        }
        Write-Host ""
    }
}

Write-Host "PIPELINE COMPLETE. All models trained and exported." -ForegroundColor Cyan
