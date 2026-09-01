@echo off
setlocal

rem 安装 Chocolatey（使用 PowerShell 执行官方安装脚本）
powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))"

rem 使用 choco 安装 ffmpeg
rem 若当前会话尚未刷新 PATH，优先尝试通过完整路径调用；否则退回到在 CMD 中执行命令
if exist "%ProgramData%\chocolatey\bin\choco.exe" (
    "%ProgramData%\chocolatey\bin\choco.exe" install ffmpeg -y
) else (
    cmd /c "choco install ffmpeg -y"
)

endlocal

