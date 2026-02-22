$command = @'
Set-Location 'C:\Users\user\Desktop\chatterbox'
& '.\.venv\Scripts\Activate.ps1'

Write-Host ''
$lineWidth = 40
$title = 'TURBO TTS'
$subtitle = 'Fast expressive generation with tags.'
$titleIndent = [Math]::Max(0, [int](($lineWidth - $title.Length) / 2))
$subtitleIndent = [Math]::Max(0, [int](($lineWidth - $subtitle.Length) / 2))

Write-Host ('=' * $lineWidth) -ForegroundColor Cyan
Write-Host ((' ' * $titleIndent) + $title) -ForegroundColor Cyan
Write-Host ((' ' * $subtitleIndent) + $subtitle) -ForegroundColor Cyan
Write-Host ('=' * $lineWidth) -ForegroundColor Cyan
Write-Host ''
Write-Host '(.venv) C:\Users\user\Desktop\chatterbox' -ForegroundColor Green

& '.\.venv\Scripts\python.exe' '.\gradio_tts_turbo_app.py'
if ($LASTEXITCODE -ne 0) {
    Write-Host ("App exited with code $LASTEXITCODE") -ForegroundColor Red
}
'@

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command", $command
)