# ═══════════════════════════════════════════════════════════════════════════════
# R-JEPA Windows Service Setup
# ═══════════════════════════════════════════════════════════════════════════════
# This script creates a Windows Service to run R-JEPA components automatically
# Uses NSSM (Non-Sucking Service Manager) for reliable service management
# ═══════════════════════════════════════════════════════════════════════════════

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("student-llm", "latent-extraction", "continuous-training", "all")]
    [string]$Service = "student-llm",

    [Parameter(Mandatory=$false)]
    [switch]$Install,

    [Parameter(Mandatory=$false)]
    [switch]$Uninstall,

    [Parameter(Mandatory=$false)]
    [switch]$Start,

    [Parameter(Mandatory=$false)]
    [switch]$Stop,

    [Parameter(Mandatory=$false)]
    [switch]$Status
)

# Configuration
$ProjectRoot = "C:\Users\teleadmin\world-txt-model"
$VenvPath = "$ProjectRoot\.venv"
$PythonExe = "$VenvPath\Scripts\python.exe"
$NSSMUrl = "https://nssm.cc/release/nssm-2.24.zip"
$NSSMPath = "$ProjectRoot\tools\nssm.exe"

# Service definitions
$Services = @{
    "student-llm" = @{
        Name = "RJEPA-StudentLLM"
        Description = "R-JEPA Student LLM Server (Qwen3-8B on GPU)"
        Script = "rjepa\llm\server.py"
        Args = "--port 8000 --model Qwen/Qwen3-8B --device cuda:0"
        LogFile = "$ProjectRoot\logs\student-llm\service.log"
    }
    "latent-extraction" = @{
        Name = "RJEPA-LatentExtraction"
        Description = "R-JEPA Continuous Latent Extraction Pipeline"
        Script = "scripts\continuous_latent_extraction.py"
        Args = "--watch-dir data\datasets\academic --output-dir data\latents"
        LogFile = "$ProjectRoot\logs\latent-extraction\service.log"
    }
    "continuous-training" = @{
        Name = "RJEPA-ContinuousTraining"
        Description = "R-JEPA Continuous Training Loop (Nightly)"
        Script = "rjepa\pipeline\continuous_training.py"
        Args = "--schedule nightly"
        LogFile = "$ProjectRoot\logs\training\service.log"
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

function Test-Admin {
    $currentUser = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    return $currentUser.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Install-NSSM {
    if (Test-Path $NSSMPath) {
        Write-Host "[OK] NSSM already installed at $NSSMPath" -ForegroundColor Green
        return
    }

    Write-Host "[INSTALL] Downloading NSSM from $NSSMUrl..." -ForegroundColor Yellow
    $TempZip = "$env:TEMP\nssm.zip"
    $TempDir = "$env:TEMP\nssm"

    Invoke-WebRequest -Uri $NSSMUrl -OutFile $TempZip
    Expand-Archive -Path $TempZip -DestinationPath $TempDir -Force

    # Find nssm.exe (architecture detection)
    if ([Environment]::Is64BitOperatingSystem) {
        $NSSMExe = Get-ChildItem -Path $TempDir -Recurse -Filter "nssm.exe" | Where-Object { $_.FullName -like "*win64*" } | Select-Object -First 1
    } else {
        $NSSMExe = Get-ChildItem -Path $TempDir -Recurse -Filter "nssm.exe" | Where-Object { $_.FullName -like "*win32*" } | Select-Object -First 1
    }

    # Create tools directory
    $ToolsDir = "$ProjectRoot\tools"
    if (-not (Test-Path $ToolsDir)) {
        New-Item -ItemType Directory -Path $ToolsDir -Force | Out-Null
    }

    # Copy nssm.exe
    Copy-Item -Path $NSSMExe.FullName -Destination $NSSMPath -Force

    # Cleanup
    Remove-Item -Path $TempZip -Force
    Remove-Item -Path $TempDir -Recurse -Force

    Write-Host "[OK] NSSM installed at $NSSMPath" -ForegroundColor Green
}

function Install-Service {
    param(
        [string]$ServiceKey
    )

    $ServiceConfig = $Services[$ServiceKey]
    $ServiceName = $ServiceConfig.Name

    # Check if service already exists
    $ExistingService = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if ($ExistingService) {
        Write-Host "[SKIP] Service $ServiceName already exists. Use -Uninstall first." -ForegroundColor Yellow
        return
    }

    Write-Host "[INSTALL] Creating service $ServiceName..." -ForegroundColor Cyan

    # Create log directory
    $LogDir = Split-Path $ServiceConfig.LogFile -Parent
    if (-not (Test-Path $LogDir)) {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    }

    # Install service with NSSM
    $ScriptPath = "$ProjectRoot\$($ServiceConfig.Script)"

    # Verify Python exists
    if (-not (Test-Path $PythonExe)) {
        Write-Host "[ERROR] Python not found at: $PythonExe" -ForegroundColor Red
        Write-Host "        Run: python -m venv .venv" -ForegroundColor Yellow
        return
    }

    # Install with full quoted paths
    & $NSSMPath install $ServiceName "`"$PythonExe`""
    & $NSSMPath set $ServiceName AppDirectory "`"$ProjectRoot`""
    & $NSSMPath set $ServiceName AppParameters "`"$ScriptPath`" $($ServiceConfig.Args)"
    & $NSSMPath set $ServiceName DisplayName $ServiceConfig.Description
    & $NSSMPath set $ServiceName Description $ServiceConfig.Description
    & $NSSMPath set $ServiceName Start SERVICE_AUTO_START

    # Logging
    & $NSSMPath set $ServiceName AppStdout $ServiceConfig.LogFile
    & $NSSMPath set $ServiceName AppStderr $ServiceConfig.LogFile

    # Environment (important for CUDA)
    & $NSSMPath set $ServiceName AppEnvironmentExtra "CUDA_VISIBLE_DEVICES=0"

    Write-Host "[OK] Service $ServiceName installed" -ForegroundColor Green
    Write-Host "     Log file: $($ServiceConfig.LogFile)" -ForegroundColor Gray
}

function Uninstall-Service {
    param(
        [string]$ServiceKey
    )

    $ServiceConfig = $Services[$ServiceKey]
    $ServiceName = $ServiceConfig.Name

    # Check if service exists
    $ExistingService = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if (-not $ExistingService) {
        Write-Host "[SKIP] Service $ServiceName does not exist" -ForegroundColor Yellow
        return
    }

    Write-Host "[UNINSTALL] Removing service $ServiceName..." -ForegroundColor Cyan

    # Stop service if running
    if ($ExistingService.Status -eq "Running") {
        Write-Host "  Stopping service..." -ForegroundColor Yellow
        & $NSSMPath stop $ServiceName
        Start-Sleep -Seconds 2
    }

    # Remove service
    & $NSSMPath remove $ServiceName confirm

    Write-Host "[OK] Service $ServiceName removed" -ForegroundColor Green
}

function Start-ServiceWrapper {
    param(
        [string]$ServiceKey
    )

    $ServiceConfig = $Services[$ServiceKey]
    $ServiceName = $ServiceConfig.Name

    Write-Host "[START] Starting service $ServiceName..." -ForegroundColor Cyan

    Start-Service -Name $ServiceName
    Start-Sleep -Seconds 2

    $ServiceStatus = Get-Service -Name $ServiceName
    if ($ServiceStatus.Status -eq "Running") {
        Write-Host "[OK] Service $ServiceName is running" -ForegroundColor Green
        Write-Host "     Logs: tail -f $($ServiceConfig.LogFile)" -ForegroundColor Gray
    } else {
        Write-Host "[ERROR] Service $ServiceName failed to start" -ForegroundColor Red
        Write-Host "        Check logs: $($ServiceConfig.LogFile)" -ForegroundColor Red
    }
}

function Stop-ServiceWrapper {
    param(
        [string]$ServiceKey
    )

    $ServiceConfig = $Services[$ServiceKey]
    $ServiceName = $ServiceConfig.Name

    Write-Host "[STOP] Stopping service $ServiceName..." -ForegroundColor Cyan

    Stop-Service -Name $ServiceName -Force
    Start-Sleep -Seconds 2

    $ServiceStatus = Get-Service -Name $ServiceName
    if ($ServiceStatus.Status -eq "Stopped") {
        Write-Host "[OK] Service $ServiceName stopped" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Service $ServiceName failed to stop" -ForegroundColor Red
    }
}

function Get-ServiceStatus {
    param(
        [string]$ServiceKey
    )

    $ServiceConfig = $Services[$ServiceKey]
    $ServiceName = $ServiceConfig.Name

    $ExistingService = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if (-not $ExistingService) {
        Write-Host "[$ServiceName] NOT INSTALLED" -ForegroundColor Yellow
        return
    }

    $Status = $ExistingService.Status
    $Color = if ($Status -eq "Running") { "Green" } else { "Yellow" }

    Write-Host "[$ServiceName] $Status" -ForegroundColor $Color
    Write-Host "  Description: $($ServiceConfig.Description)" -ForegroundColor Gray
    Write-Host "  Log file: $($ServiceConfig.LogFile)" -ForegroundColor Gray
}

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "R-JEPA Windows Service Manager" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Check admin rights
if (-not (Test-Admin)) {
    Write-Host "[ERROR] This script requires Administrator privileges" -ForegroundColor Red
    Write-Host "        Right-click PowerShell and 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

# Check Python venv
if (-not (Test-Path $PythonExe)) {
    Write-Host "[ERROR] Python venv not found at $VenvPath" -ForegroundColor Red
    Write-Host "        Run: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

# Install NSSM if needed
Install-NSSM

# Determine which services to operate on
$ServiceKeys = if ($Service -eq "all") { $Services.Keys } else { @($Service) }

# Execute action
if ($Install) {
    foreach ($Key in $ServiceKeys) {
        Install-Service -ServiceKey $Key
    }
    Write-Host ""
    Write-Host "[NEXT STEPS]" -ForegroundColor Cyan
    Write-Host "  Start service: .\scripts\setup_windows_service.ps1 -Service $Service -Start" -ForegroundColor Gray
    Write-Host "  Check status: .\scripts\setup_windows_service.ps1 -Service $Service -Status" -ForegroundColor Gray
}
elseif ($Uninstall) {
    foreach ($Key in $ServiceKeys) {
        Uninstall-Service -ServiceKey $Key
    }
}
elseif ($Start) {
    foreach ($Key in $ServiceKeys) {
        Start-ServiceWrapper -ServiceKey $Key
    }
}
elseif ($Stop) {
    foreach ($Key in $ServiceKeys) {
        Stop-ServiceWrapper -ServiceKey $Key
    }
}
elseif ($Status) {
    foreach ($Key in $ServiceKeys) {
        Get-ServiceStatus -ServiceKey $Key
    }
}
else {
    Write-Host "USAGE:" -ForegroundColor Yellow
    Write-Host "  Install:   .\scripts\setup_windows_service.ps1 -Service student-llm -Install" -ForegroundColor Gray
    Write-Host "  Start:     .\scripts\setup_windows_service.ps1 -Service student-llm -Start" -ForegroundColor Gray
    Write-Host "  Stop:      .\scripts\setup_windows_service.ps1 -Service student-llm -Stop" -ForegroundColor Gray
    Write-Host "  Status:    .\scripts\setup_windows_service.ps1 -Service student-llm -Status" -ForegroundColor Gray
    Write-Host "  Uninstall: .\scripts\setup_windows_service.ps1 -Service student-llm -Uninstall" -ForegroundColor Gray
    Write-Host ""
    Write-Host "SERVICES:" -ForegroundColor Yellow
    foreach ($Key in $Services.Keys) {
        Write-Host "  - $Key" -ForegroundColor Gray
    }
    Write-Host ""
    Write-Host "  Use -Service all to manage all services" -ForegroundColor Gray
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
