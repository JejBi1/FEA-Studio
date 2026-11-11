<#
.SYNOPSIS
  Instalace FEA Studia (Windows x64) po modulech. Spouští se z rozbaleného balíčku:
    "Install FEA Studio.bat" [--profile minimal|standard|full] [modul ...]
  nebo  powershell -ExecutionPolicy Bypass -File installer\install.ps1 [parametry instalátoru]

.DESCRIPTION
  Tenký spouštěč: najde Python 3.9+ (py / python), jinak stáhne správce balíčků uv (jeden soubor ~30 MB) a jeho
  Pythonem spustí installer\fea_setup.py, který vytvoří .venv s Pythonem 3.11, nainstaluje vybrané moduly
  (fea_core\modules.json), CalculiX a vytvoří zástupce. Bez parametrů se zeptá na rozsah (minimal / standard / full).
  Podrobnosti: installer\README.md. Další moduly později: python -m fea_core.modules install <modul>
#>
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$env:PYTHONIOENCODING = "utf-8"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$Root = Split-Path -Parent $PSScriptRoot
$Setup = Join-Path $PSScriptRoot "fea_setup.py"
if (-not (Test-Path (Join-Path $Root "fea_core\__init__.py"))) {
    throw "Ve složce $Root není FEA Studio (chybí fea_core). Spusťte instalaci z rozbaleného balíčku."
}
# první parametr smí být příkaz instalátoru (install / update / uninstall …), jinak install
$commands = @("install", "update", "remove", "list", "doctor", "uninstall", "shortcuts", "fusion", "slim", "mpi")
if ($args.Count -gt 0 -and $commands -contains [string]$args[0]) { $setupArgs = @($args) } else { $setupArgs = @("install") + $args }

# mezipaměť uv uvnitř složky programu (instalátor ji na konci smaže); FEA_UV_CACHE_DIR = sdílená mezipaměť (testy)
if ($env:FEA_UV_CACHE_DIR) { $env:UV_CACHE_DIR = $env:FEA_UV_CACHE_DIR } else { $env:UV_CACHE_DIR = Join-Path $Root "tools\_download\uv-cache" }
# uv nesmí zapisovat mimo složku programu: bez spouštěče python3.11.exe v %USERPROFILE%\.local\bin a bez registrace
# Pythonu do registru (PEP 514) – platí pro `uv python install` i pro automatické stažení Pythonu (uv run / uv venv)
$env:UV_PYTHON_INSTALL_BIN = "0"
$env:UV_PYTHON_INSTALL_REGISTRY = "0"
$uvBundled = Join-Path $Root "tools\uv\uv.exe"

# setup.exe přibalí uv: instalátor pak běží na Pythonu 3.11 od uv (stejný jako program) bez ohledu na Python v systému
if (Test-Path $uvBundled) {
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $Root "tools\python"
    $env:UV_PYTHON_PREFERENCE = "only-managed"
    $log = Join-Path $Root "install.log"
    $cs = (Get-UICulture).TwoLetterISOLanguageName -eq "cs"
    # Python 3.11 zvlášť a s opakováním: antivir / synchronizace složky umí krátce zamknout čerstvě rozbalené soubory
    $err = Join-Path $Root "tools\uv\python-install.err"
    Write-Host "Python 3.11 (uv)…"
    # setup.exe / zip přibalí Python 3.11 rozbalený v tools\python\cpython-3.11.*-windows-x86_64-none → nic se nestahuje
    $pyOk = [bool](Get-ChildItem -Path (Join-Path $env:UV_PYTHON_INSTALL_DIR "cpython-3.11.*-windows-x86_64-none\python.exe") -ErrorAction SilentlyContinue)
    if ($pyOk) { Write-Host "Python 3.11: přibalený / bundled" }
    for ($i = 1; $i -le 2 -and -not $pyOk; $i++) {
        # druhý pokus s podrobným výpisem uv (cesty dočasných složek) do install.log
        $uvArgs = @("python", "install", "3.11", "--no-bin", "--no-registry"); if ($i -eq 2) { $uvArgs += "-v" }
        $p = Start-Process -FilePath $uvBundled -ArgumentList $uvArgs -NoNewWindow -Wait -PassThru -RedirectStandardError $err
        if ($p.ExitCode -eq 0) { $pyOk = $true; break }
        $msg = "install.ps1: uv python install 3.11 -> kód $($p.ExitCode) (pokus $i): " + (Get-Content $err -Raw -ErrorAction SilentlyContinue)
        if ($i -eq 1) { Write-Host $msg }
        Add-Content -Path $log -Value $msg -Encoding UTF8
        Start-Sleep -Seconds 3
    }
    Remove-Item $err -Force -ErrorAction SilentlyContinue
    if (-not $pyOk) {
        # uv rozbaluje do tools\python\.temp\.tmpXXXX – u některých antivirů to selže („Failed to extract archive …
        # os error 3“, 20. 10. 2026). Náhrada: stejné sestavení Pythonu stažené přímo (otisk = PY_SHA256 v fea_setup.py)
        # a rozbalené tar.exe z Windows do složky, kterou uv pozná jako spravovaný Python.
        if ($cs) { Write-Host "uv python install selhal – stahuji Python 3.11.17 přímo…" -ForegroundColor Yellow }
        else { Write-Host "uv python install failed – downloading Python 3.11.17 directly…" -ForegroundColor Yellow }
        $pyDir = $env:UV_PYTHON_INSTALL_DIR
        $target = Join-Path $pyDir "cpython-3.11.17-windows-x86_64-none"
        $tmp = Join-Path $pyDir "_fea_extract"
        $dl = Join-Path $Root "tools\_download"
        $arc = Join-Path $dl "cpython-3.11.17-windows-x86_64.tar.gz"
        try {
            New-Item -ItemType Directory -Force -Path $dl, $pyDir | Out-Null
            $url = "https://github.com/astral-sh/python-build-standalone/releases/download/20261013/cpython-3.11.17%2B20261013-x86_64-pc-windows-msvc-install_only_stripped.tar.gz"
            $sha = "54a38c5e7cc1a5fd2b0182b495f29606dc1455d18ebfd3b5bf0a3c09253b5031"
            if (-not ((Test-Path $arc) -and (Get-FileHash -Algorithm SHA256 $arc).Hash.ToLower() -eq $sha)) {
                Invoke-WebRequest -Uri $url -OutFile $arc -UseBasicParsing
            }
            $got = (Get-FileHash -Algorithm SHA256 $arc).Hash.ToLower()
            if ($got -ne $sha) { Remove-Item $arc -Force; throw "SHA-256 $got ≠ $sha" }
            foreach ($d in @($tmp, $target)) { if (Test-Path $d) { Remove-Item $d -Recurse -Force } }
            New-Item -ItemType Directory -Force -Path $tmp | Out-Null
            & (Join-Path $env:SystemRoot "System32\tar.exe") -xzf $arc -C $tmp
            if ($LASTEXITCODE -ne 0) { throw "tar.exe -> $LASTEXITCODE" }
            Move-Item (Join-Path $tmp "python") $target
            Remove-Item $tmp -Recurse -Force
            if (-not (Test-Path (Join-Path $target "python.exe"))) { throw "python.exe chybí / missing" }
            Add-Content -Path $log -Value "install.ps1: Python 3.11.17 stažen přímo -> $target" -Encoding UTF8
        } catch {
            Add-Content -Path $log -Value "install.ps1: přímá instalace Pythonu selhala: $_" -Encoding UTF8
            Write-Host ""
            if ($cs) {
                Write-Host "Python 3.11 se nepodařilo nainstalovat do $pyDir ($_)." -ForegroundColor Red
                Write-Host "Nejčastější příčina: antivir maže / blokuje čerstvě rozbalené soubory – přidejte výjimku pro složku" -ForegroundColor Yellow
                Write-Host "$Root a spusťte instalaci znovu. Podrobnosti: $log" -ForegroundColor Yellow
            } else {
                Write-Host "Python 3.11 could not be installed to $pyDir ($_)." -ForegroundColor Red
                Write-Host "Most common cause: an antivirus deletes / blocks freshly extracted files – add an exclusion for" -ForegroundColor Yellow
                Write-Host "$Root and run the installation again. Details: $log" -ForegroundColor Yellow
            }
            exit 3
        }
    }
    $before = 0; if (Test-Path $log) { $before = (Get-Item $log).Length }
    & $uvBundled run --no-project --no-config --python 3.11 $Setup @setupArgs
    $code = $LASTEXITCODE
    $after = 0; if (Test-Path $log) { $after = (Get-Item $log).Length }
    if ($code -ne 0 -and $after -eq $before) {
        # fea_setup.py se vůbec nespustil (chyba uv) – druhý pokus s chybovým výstupem do install.log
        $uvArgs = @("run", "--no-project", "--no-config", "--python", "3.11", "`"$Setup`"") + ($setupArgs | ForEach-Object { "`"$_`"" })
        $p = Start-Process -FilePath $uvBundled -ArgumentList $uvArgs -NoNewWindow -Wait -PassThru -RedirectStandardError $err
        $code = $p.ExitCode
        if ($code -ne 0) { Add-Content -Path $log -Value ("install.ps1: uv run -> kód $code`n" + (Get-Content $err -Raw -ErrorAction SilentlyContinue)) -Encoding UTF8 }
        Remove-Item $err -Force -ErrorAction SilentlyContinue
    }
    if ($code -ne 0) { Add-Content -Path $log -Value "install.ps1: fea_setup.py $($setupArgs -join ' ') -> kód $code" -Encoding UTF8 }
    exit $code
}

function Test-Python($exe, [string[]]$pre) {
    try {
        $v = & $exe @pre -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -ne 0) { return $false }
        $p = $v.Trim().Split(".")
        return ([int]$p[0] -eq 3 -and [int]$p[1] -ge 9)
    } catch { return $false }
}

$py = $null
foreach ($c in @(@("py", "-3"), @("python"), @("python3"))) {
    if (Get-Command $c[0] -ErrorAction SilentlyContinue) {
        $pre = @(); if ($c.Count -gt 1) { $pre = $c[1..($c.Count - 1)] }
        if (Test-Python $c[0] $pre) { $py = @($c[0]) + $pre; break }
    }
}

if ($py) {
    & $py[0] @($py[1..($py.Count)] | Where-Object { $_ }) $Setup @setupArgs
    exit $LASTEXITCODE
}

# žádný použitelný Python: uv (stažený do tools\uv) spustí instalátor svým Pythonem 3.11
Write-Host "Python 3.9+ nenalezen - stahuji správce balíčků uv (Python 3.11 si stáhne sám)." -ForegroundColor Cyan
$uvDir = Join-Path $Root "tools\uv"
$uv = Join-Path $uvDir "uv.exe"
if (-not (Test-Path $uv)) {
    New-Item -ItemType Directory -Force -Path $uvDir | Out-Null
    $dl = Join-Path $Root "tools\_download"
    New-Item -ItemType Directory -Force -Path $dl | Out-Null
    $zip = Join-Path $dl "fea_uv.zip"
    # připnutá verze a otisk SHA-256 (stejné jako UV_URL / UV_SHA256 v installer\fea_setup.py)
    Invoke-WebRequest -Uri "https://github.com/astral-sh/uv/releases/download/0.13.1/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip -UseBasicParsing
    $got = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
    if ($got -ne "3b2661a463a5baea7f5076ea2b25ab71d77f36a6c0d809e5350651792f80a672") {
        Remove-Item $zip -Force
        Write-Host "CHYBA: otisk SHA-256 staženého uv nesouhlasí ($got) - instalace přerušena." -ForegroundColor Red
        exit 1
    }
    Expand-Archive -Path $zip -DestinationPath $uvDir -Force
    Remove-Item $zip -Force
}
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Root "tools\python"
$env:UV_PYTHON_PREFERENCE = "only-managed"
& $uv run --no-project --no-config --python 3.11 $Setup @setupArgs
exit $LASTEXITCODE
