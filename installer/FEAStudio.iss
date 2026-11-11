; Inno Setup 6 – instalátor FEA Studia pro Windows (setup.exe). Sestavení: python installer\make_package.py
; (ISCC z tools\innosetup – python installer\get_innosetup.py – předá AppVersion, SourceDir = rozbalený balíček, UvExe).
;
; Co dělá:
;  - zkopíruje soubory programu do {app} (výchozí %LOCALAPPDATA%\Programs\FEA Studio, bez práv správce),
;  - přibalí Python 3.11 (tools\python – make_package.bundle_python), uv.exe a CalculiX ccx_static.exe (tools\calculix;
;    ccx_i4.exe s Intel MKL se nešíří – GPL-2.0 – instalátor ho rozbalí z původního zipu dhondt.de jen s modulem pardiso,
;    potřebuje internet i u offline varianty) a spustí
;    PŘÍMO přibaleným Pythonem installer\fea_setup.py install|update s moduly vybranými v komponentách – bez PowerShellu
;    (zásady skupiny ExecutionPolicy / Constrained Language ve firmách; antiviry a EDR hlásí „powershell -ExecutionPolicy
;    Bypass“ spuštěný instalátorem jako podezřelý). install.ps1 jen záloha pro balíček bez Pythonu. Balíčky Pythonu se
;    stáhnou – offline varianta (/DOffline, make_package.py --offline) je nese v tools\offline\wheels. Nic do systému
;    kromě zástupců a přípony .fea,
;  - AKTUALIZACE: AppId je pro všechny verze stejný → nová verze se nainstaluje přes starou (stejná složka, minulé
;    komponenty předvyplněné; u instalace z .zip / .bat se moduly přečtou z .venv\fea_modules.json). fea_setup.py update
;    pak spustí migrace (installer\migrations.py), smaže zastaralé soubory podle manifestu a přesynchronizuje moduly.
;  - návrat ke starší verzi odmítne (PrepareToInstall – před kopírováním souborů), povolí ho /FORCE.
;  - ukázky examples\*.fea se přepíší novou verzí; uživatelem upravená ukázka se předtím přejmenuje na <název>.fea.bak.
;  - odinstalace: fea_setup.py uninstall (zástupci a přípona .fea jen té instalace, doplněk Fusion, bajtkód) + .venv, tools;
;    nakonec prázdné složky programu (i ty z rozbaleného zipu, které Inno nezná). Vlastní soubory zůstanou.
;
; Parametry příkazové řádky (vedle standardních /SILENT /VERYSILENT /DIR= /COMPONENTS= /TYPE= /LOG=):
;   /NOSHORTCUTS /NOMCP /NOFUSION /NOCHECK /QUICKCHECK /FORCE  → --no-shortcuts --no-mcp --no-fusion --no-check …
; Návratový kód: 0 = v pořádku; 7 = odmítnutý návrat ke starší verzi; 20 + kód fea_setup.py = selhala instalace modulů.

; verze = fea_core.__version__, předává ji make_package.py (/DAppVersion=…); bez ní se nesestaví (dřív zastaralá výchozí hodnota)
#ifndef AppVersion
  #error AppVersion is not defined – build setup.exe with python installer/make_package.py (passes /DAppVersion from fea_core.__version__)
#endif
#ifndef SourceDir
  #define SourceDir ".."
#endif
; složka přibaleného Pythonu (fea_setup.PY_DIR) – předává make_package.py
#ifndef PyDir
  #define PyDir "cpython-3.11.17-windows-x86_64-none"
#endif
#ifdef Offline
  #define OutName "FEAStudio-" + AppVersion + "-offline-setup"
#else
  #define OutName "FEAStudio-" + AppVersion + "-setup"
#endif

[Setup]
; stejný AppId pro všechny verze – aktualizace přes starou instalaci (NEMĚNIT)
AppId={{6F1E7C2A-6E0B-4B7B-9C55-FEA5D1000001}
AppName=FEA Studio
AppVersion={#AppVersion}
AppVerName=FEA Studio {#AppVersion}
AppPublisher=FEA Studio
AppCopyright=Copyright (C) 2026 FEA Studio authors, GPL-3.0-or-later
VersionInfoVersion={#AppVersion}
VersionInfoProductName=FEA Studio
VersionInfoProductVersion={#AppVersion}
VersionInfoCompany=FEA Studio
VersionInfoDescription=FEA Studio {#AppVersion} – instalátor / setup
; Windows 10 1809+ (Qt 6, Python 3.11, tar.exe a curl v systému); starší systém skončí hned srozumitelnou hláškou
MinVersion=10.0.17763
DefaultDirName={localappdata}\Programs\FEA Studio
UsePreviousAppDir=yes
; typ instalace a komponenty z minulé instalace (Inno je ukládá do odinstalačního záznamu)
UsePreviousSetupType=yes
DisableDirPage=auto
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputBaseFilename={#OutName}
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName=FEA Studio
UninstallDisplayIcon={app}\installer\icons\fea_studio.ico
SetupIconFile={#SourceDir}\installer\icons\fea_studio.ico
SetupLogging=yes
; RedirectionGuard (Inno 6.5+, výchozí zapnuto) zdědí i spuštěné procesy: uv dělá v tools\python spojení (junction)
; cpython-3.11-… → „os error 448: cesta obsahuje nedůvěryhodný přípojný bod“ a instalace modulů selže (ověřeno 15. 10. 2026).
; Instalace běží bez práv správce do profilu uživatele → ochrana proti eskalaci práv tu nemá co chránit.
RedirectionGuard=no
ShowLanguageDialog=yes

[Languages]
Name: "cs"; MessagesFile: "compiler:Languages\Czech.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
cs.TypeStandard=Standardní – GUI, únava, 3D tisk, MKL PARDISO, prutové průřezy, protokol Word, mechanismy
en.TypeStandard=Standard – GUI, fatigue, 3D printing, MKL PARDISO, beam sections, Word report, mechanisms
cs.TypeMinimal=Minimální – jen jádro MKP (STEP, síť, CalculiX, příkazová řádka)
en.TypeMinimal=Minimal – FEM core only (STEP, mesh, CalculiX, command line)
cs.TypeFull=Úplná – navíc MCP / REST, řešič velkých modelů (NGSolve + hypre)
en.TypeFull=Full – plus MCP / REST, large-model solver (NGSolve + hypre)
cs.TypeCustom=Vlastní výběr modulů
en.TypeCustom=Custom selection of modules
cs.Installing=Stahuji a instaluji vybrané moduly (několik minut)…
en.Installing=Downloading and installing the selected modules (several minutes)…
cs.SetupFailed=Instalace modulů skončila chybou (kód %1). Podrobnosti jsou v souboru install.log ve složce programu; instalaci lze zopakovat spuštěním „Install FEA Studio.bat“ v téže složce.
en.SetupFailed=Installing the modules failed (code %1). See install.log in the program folder; you can repeat the installation by running “Install FEA Studio.bat” in that folder.
cs.Downgrade=Ve složce %1 je nainstalována novější verze FEA Studia (%2) než tento instalátor (%3). Návrat ke starší verzi: spusťte instalátor s parametrem /FORCE.
en.Downgrade=A newer version of FEA Studio (%2) is installed in %1 than this installer (%3). To downgrade, run the installer with /FORCE.
cs.PathTooLong=Cesta %1 je pro FEA Studio příliš dlouhá (%2 znaků, nejvýš 100): soubory knihoven by přesáhly limit 260 znaků Windows. Zvolte kratší cestu (např. C:\FEA Studio), nebo zapněte dlouhé cesty (zásada „Povolit dlouhé cesty Win32“, LongPathsEnabled = 1).
en.PathTooLong=The path %1 is too long for FEA Studio (%2 characters, at most 100): library files would exceed the 260-character Windows limit. Choose a shorter path (e.g. C:\FEA Studio), or enable long paths (policy “Enable Win32 long paths”, LongPathsEnabled = 1).
cs.NetworkPath=FEA Studio nejde nainstalovat na síťovou cestu %1 (Python a řešiče se z ní spouštějí pomalu a antiviry / zásady je často blokují). Zvolte místní disk.
en.NetworkPath=FEA Studio cannot be installed to the network path %1 (Python and the solvers would run slowly from it and antiviruses / policies often block them). Choose a local disk.
cs.Running=Z instalační složky FEA Studia běží programy, které drží soubory instalace:%n%n%1%nZavřete okno FEA Studia a ukončete klienty AI, které spouštějí MCP server fea-studio (Claude Desktop, Claude Code, Cursor, VS Code…), pak klikněte na Zpět a znovu na Instalovat.
en.Running=Programs from the FEA Studio folder are running and lock installed files:%n%n%1%nClose the FEA Studio window and quit AI clients that run the fea-studio MCP server (Claude Desktop, Claude Code, Cursor, VS Code…), then click Back and Install again.

[Types]
Name: "standard"; Description: "{cm:TypeStandard}"
Name: "minimal"; Description: "{cm:TypeMinimal}"
Name: "full"; Description: "{cm:TypeFull}"
Name: "custom"; Description: "{cm:TypeCustom}"; Flags: iscustom

[Components]
Name: "core"; Description: "Jádro MKP / FEM core (~700 MB)"; Types: minimal standard full custom; ExtraDiskSpaceRequired: 734003200; Flags: fixed
Name: "gui"; Description: "Grafické rozhraní / Graphical interface (~80 MB)"; Types: standard full; ExtraDiskSpaceRequired: 83886080
Name: "plots"; Description: "Grafy a obrázky / Plots – matplotlib (~45 MB)"; Types: standard full; ExtraDiskSpaceRequired: 47185920
Name: "accel"; Description: "Numba: únava, 3D tisk, obrázky / fatigue, 3D printing, images (~130 MB)"; Types: standard full; ExtraDiskSpaceRequired: 136314880
Name: "pardiso"; Description: "Rychlý řešič / Fast solver PARDISO – Intel MKL (~400 MB)"; Types: standard full; ExtraDiskSpaceRequired: 419430400
Name: "beams"; Description: "Vlastní průřezy prutů / Custom beam sections – sectionproperties (~15 MB)"; Types: standard full; ExtraDiskSpaceRequired: 15728640
Name: "report"; Description: "Protokol Word / Word report (~11 MB)"; Types: standard full; ExtraDiskSpaceRequired: 11534336
Name: "motion"; Description: "Mechanismy / Mechanisms – Exudyn (~22 MB)"; Types: standard full; ExtraDiskSpaceRequired: 23068672
Name: "mcp"; Description: "Server MCP a REST API / MCP server and REST API (~42 MB)"; Types: full; ExtraDiskSpaceRequired: 44040192
Name: "ngsolve"; Description: "Řešič NGSolve / NGSolve solver (~180 MB)"; Types: full; ExtraDiskSpaceRequired: 184549376
Name: "hypre"; Description: "Řešič velkých modelů / Large-model solver hypre + MS-MPI (~70 MB, MS-MPI: admin)"; Types: full; ExtraDiskSpaceRequired: 75497472
Name: "chrono"; Description: "Kontakt v mechanismu / Contact in mechanisms – Project Chrono (~600 MB)"; Types: custom; ExtraDiskSpaceRequired: 629145600
Name: "native"; Description: "Nativní soubory / Native files – Inventor .ipt, Fusion .f3d (<1 MB)"; Types: custom; ExtraDiskSpaceRequired: 1048576
Name: "verify"; Description: "Ověřování a testy / Verification and tests – MYSTRAN, pytest (~50 MB)"; Types: custom; ExtraDiskSpaceRequired: 53477376

[InstallDelete]
; manifest souborů přijde s novou verzí – starý se nesmí splést s novým (balíček bez manifestu = nic se nemaže)
Type: files; Name: "{app}\installer\manifest.json"
Type: filesandordirs; Name: "{app}\installer\__pycache__"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Excludes: "\examples\*.fea,\tools\python\*"; Flags: recursesubdirs createallsubdirs ignoreversion
; přibalený Python (složka nese celou verzi cpython-3.11.17-…) se jen doplní, nepřepisuje: při aktualizaci ho drží
; běžící procesy staré instalace (MCP server fea-studio spuštěný klientem AI, GUI) → „DeleteFile selhala; kód 5“
; u python3.dll; obsah stejně pojmenované složky je bajtově stejný
Source: "{#SourceDir}\tools\python\*"; DestDir: "{app}\tools\python"; \
  Flags: recursesubdirs createallsubdirs onlyifdoesntexist skipifsourcedoesntexist
; ukázky zvlášť: uživatelem upravenou ukázku (otisk nesedí s žádnou vydanou verzí) před přepsáním přejmenuje na
; <název>.fea.bak (BackupModifiedExample); nezměněné se přepíší – nové verze ukázek se tak vždy aktualizují
Source: "{#SourceDir}\examples\*.fea"; DestDir: "{app}\examples"; Flags: ignoreversion skipifsourcedoesntexist; \
  BeforeInstall: BackupModifiedExample
; manifest nové verze jen pro [Code] (otisky ukázek) – rozbalí se do {tmp}, do {app} jde běžnou cestou výše
Source: "{#SourceDir}\installer\manifest.json"; DestName: "fea_new_manifest.json"; Flags: dontcopy skipifsourcedoesntexist
#ifdef UvExe
Source: "{#UvExe}"; DestDir: "{app}\tools\uv"; Flags: ignoreversion
#endif

[UninstallRun]
Filename: "{app}\.venv\Scripts\python.exe"; Parameters: """{app}\installer\fea_setup.py"" uninstall --quiet --pycache"; \
  WorkingDir: "{app}"; Flags: runhidden waituntilterminated skipifdoesntexist; RunOnceId: "FEAStudioUninstall"

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\tools"
Type: files; Name: "{app}\install.log"
; výstup kontrolního výpočtu instalátoru (fea_core.warmup)
Type: filesandordirs; Name: "{app}\verification\out"

[Code]
const
  { volitelné moduly = komponenty (core je vždy) }
  MODULES = 'gui plots accel pardiso beams report motion mcp ngsolve hypre chrono native verify';
  FA_DIRECTORY = $10;
  FA_REPARSE_POINT = $400;

var
  InstallResult: Integer;
  PreviousInno: Boolean;
  { manifesty souborů programu (installer/upgrade.py: řádky "cesta": "otisk SHA-256[:16]") – uložený minulou instalací,
    manifest balíčku ve složce a manifest nové verze; podle nich BackupModifiedExample pozná nezměněnou ukázku }
  KnownManifests: String;

function HasSwitch(const Name: String): Boolean;
var
  i: Integer;
begin
  Result := False;
  for i := 1 to ParamCount do
    if CompareText(ParamStr(i), Name) = 0 then
      Result := True;
end;

{ verze „1.2.3“ → číslo pro porovnání }
function VersionNumber(V: String): Integer;
var
  i, part, n: Integer;
begin
  Result := 0;
  part := 0;
  n := 0;
  V := V + '.';
  for i := 1 to Length(V) do
  begin
    if (V[i] >= '0') and (V[i] <= '9') then
      part := part * 10 + Ord(V[i]) - Ord('0')
    else if V[i] = '.' then
    begin
      if n < 3 then
        Result := Result * 1000 + part;
      n := n + 1;
      part := 0;
    end
    else
      Break;
  end;
  while n < 3 do
  begin
    Result := Result * 1000;
    n := n + 1;
  end;
end;

function StateFile(const Dir: String): String;
begin
  Result := AddBackslash(Dir) + '.venv\fea_modules.json';
end;

{ verze ze stavu instalace (.venv\fea_modules.json); stav bez verze = 1.0.0; bez stavu = '' }
function InstalledVersion(const Dir: String): String;
var
  S: AnsiString;
  p: Integer;
begin
  Result := '';
  if not LoadStringFromFile(StateFile(Dir), S) then
    Exit;
  Result := '1.0.0';
  p := Pos('"version": "', S);
  if p > 0 then
  begin
    S := Copy(S, p + 12, 40);
    p := Pos('"', S);
    if p > 1 then
      Result := Copy(S, 1, p - 1);
  end;
end;

function InitializeSetup(): Boolean;
begin
  PreviousInno := RegKeyExists(HKCU, 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{6F1E7C2A-6E0B-4B7B-9C55-FEA5D1000001}_is1');
  InstallResult := 0;
  Result := True;
end;

{ instalace z .zip / .bat bez záznamu Inno: komponenty podle modulů v .venv\fea_modules.json }
procedure CurPageChanged(CurPageID: Integer);
var
  S: AnsiString;
  Sel, Rest, Name: String;
  p: Integer;
begin
  if (CurPageID = wpSelectComponents) and (not PreviousInno) and LoadStringFromFile(StateFile(WizardDirValue), S) then
  begin
    Sel := '';
    Rest := MODULES + ' ';
    p := Pos(' ', Rest);
    while p > 0 do
    begin
      Name := Copy(Rest, 1, p - 1);
      Rest := Copy(Rest, p + 1, Length(Rest));
      if Sel <> '' then
        Sel := Sel + ',';
      if Pos('"' + Name + '"', S) > 0 then
        Sel := Sel + Name
      else
        Sel := Sel + '!' + Name;
      p := Pos(' ', Rest);
    end;
    WizardSelectComponents(Sel);
  end;
end;

procedure LoadManifests(const Dir: String);
var
  S: AnsiString;
begin
  KnownManifests := '';
  if LoadStringFromFile(AddBackslash(Dir) + '.venv\fea_install_manifest.json', S) then
    KnownManifests := KnownManifests + S;
  { manifest balíčku ve složce smaže [InstallDelete] – načíst ještě před instalací (PrepareToInstall) }
  if LoadStringFromFile(AddBackslash(Dir) + 'installer\manifest.json', S) then
    KnownManifests := KnownManifests + S;
  try
    ExtractTemporaryFile('fea_new_manifest.json');
    if LoadStringFromFile(ExpandConstant('{tmp}\fea_new_manifest.json'), S) then
      KnownManifests := KnownManifests + S;
  except
    Log('fea_setup: manifest nové verze nejde načíst: ' + GetExceptionMessage);
  end;
end;

{ BeforeInstall ukázek examples\*.fea: ukázka, jejíž otisk nesedí s žádnou vydanou podobou (upravil ji uživatel), se před
  přepsáním přejmenuje na <název>.fea.bak (.bak2, .bak3… – starší zálohy se nepřepisují). Nezměněná ukázka se přepíše bez
  zálohy. Bez známého manifestu (instalace bez seznamu souborů) se zálohuje každá odlišná ukázka. }
procedure BackupModifiedExample();
var
  Dest, Digest, Bak: String;
  i: Integer;
begin
  Dest := ExpandConstant(CurrentFileName);
  if not FileExists(Dest) then
    Exit;
  try
    Digest := Copy(LowerCase(GetSHA256OfFile(Dest)), 1, 16);
  except
    Log('fea_setup: otisk ukázky ' + Dest + ' nejde zjistit: ' + GetExceptionMessage);
    Exit;
  end;
  if Pos('"examples/' + ExtractFileName(Dest) + '": "' + Digest + '"', KnownManifests) > 0 then
    Exit;
  Bak := Dest + '.bak';
  i := 2;
  while FileExists(Bak) and (i < 1000) do
  begin
    Bak := Dest + '.bak' + IntToStr(i);
    i := i + 1;
  end;
  if RenameFile(Dest, Bak) then
    Log('fea_setup: upravená ukázka zálohována: ' + Bak)
  else
    Log('fea_setup: zálohu upravené ukázky ' + Dest + ' nejde vytvořit');
end;

{ Procesy spuštěné ze složky instalace (GUI, MCP server fea-studio od klienta AI, výpočty) drží python3.dll a balíčky
  v .venv – přepsání by selhalo uprostřed instalace („DeleteFile selhala; kód 5“). Seznam „název (PID)“, '' = nic. }
function RunningFromDir(Dir: String): String;
var
  Locator, Service, Procs, Item: Variant;
  i, Pid: Integer;
  Path, Name: String;
begin
  Result := '';
  Dir := Lowercase(AddBackslash(Dir));
  try
    Locator := CreateOleObject('WbemScripting.SWbemLocator');
    Service := Locator.ConnectServer('.', 'root\CIMV2');
    Procs := Service.ExecQuery('SELECT ProcessId, Name, ExecutablePath FROM Win32_Process');
    for i := 0 to Procs.Count - 1 do
    begin
      Item := Procs.ItemIndex(i);
      if not VarIsNull(Item.ExecutablePath) then
      begin
        Path := Item.ExecutablePath;
        Name := Item.Name;
        Pid := Item.ProcessId;
        if Pos(Dir, Lowercase(Path)) = 1 then
          Result := Result + '  ' + Name + ' (PID ' + IntToStr(Pid) + ')' + #13#10;
      end;
    end;
  except
    Result := '';                                   { WMI nedostupné → bez kontroly (jako dřív) }
  end;
end;

{ dlouhé cesty Windows (LongPathsEnabled) – bez nich se v .venv nevejdou nejdelší cesty knihoven (~150 znaků) }
function LongPathsEnabled(): Boolean;
var
  V: Cardinal;
begin
  Result := RegQueryDWordValue(HKLM, 'SYSTEM\CurrentControlSet\Control\FileSystem', 'LongPathsEnabled', V) and (V = 1);
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Old, Running: String;
begin
  Result := '';
  Old := InstalledVersion(WizardDirValue);
  if Copy(WizardDirValue, 1, 2) = '\\' then
    Result := FmtMessage(CustomMessage('NetworkPath'), [WizardDirValue])
  else if (Length(WizardDirValue) > 100) and not LongPathsEnabled() then
    Result := FmtMessage(CustomMessage('PathTooLong'), [WizardDirValue, IntToStr(Length(WizardDirValue))])
  else if (Old <> '') and (VersionNumber(Old) > VersionNumber('{#AppVersion}')) and (not HasSwitch('/FORCE')) then
    Result := FmtMessage(CustomMessage('Downgrade'), [WizardDirValue, Old, '{#AppVersion}'])
  else
  begin
    Running := RunningFromDir(WizardDirValue);
    if Running <> '' then
      Result := FmtMessage(CustomMessage('Running'), [Running])
    else
      LoadManifests(WizardDirValue);
  end;
end;

function SelectedModules(): String;
var
  Rest, Name: String;
  p: Integer;
begin
  Result := 'core';
  Rest := MODULES + ' ';
  p := Pos(' ', Rest);
  while p > 0 do
  begin
    Name := Copy(Rest, 1, p - 1);
    Rest := Copy(Rest, p + 1, Length(Rest));
    if WizardIsComponentSelected(Name) then
      Result := Result + ' ' + Name;
    p := Pos(' ', Rest);
  end;
end;

function SetupArgs(): String;
begin
  if FileExists(StateFile(ExpandConstant('{app}'))) then
    Result := 'update'
  else
    Result := 'install';
  Result := Result + ' --quiet ' + SelectedModules();
  { jazyk průvodce = jazyk programu (language.txt v nastavení uživatele; v programu lze přepnout) }
  Result := Result + ' --lang ' + ActiveLanguage;
  { komponenty z minulé instalace přes setup.exe = celý výběr (odebrané moduly se odinstalují); jinak se jen přidávají }
  if PreviousInno then
    Result := Result + ' --exact';
  { komponenta hypre = souhlas s instalací Microsoft MPI (UAC); bez --msmpi ho fea_setup.py s --quiet přeskočí }
  if WizardIsComponentSelected('hypre') and not WizardSilent then
    Result := Result + ' --msmpi';
  if HasSwitch('/NOSHORTCUTS') then Result := Result + ' --no-shortcuts';
  if HasSwitch('/NOMCP') then Result := Result + ' --no-mcp';
  if HasSwitch('/NOFUSION') then Result := Result + ' --no-fusion';
  if HasSwitch('/NOCHECK') then Result := Result + ' --no-check';
  if HasSwitch('/QUICKCHECK') then Result := Result + ' --quick-check';
  if HasSwitch('/FORCE') then Result := Result + ' --force';
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Exe, Params: String;
  ShowCmd, Code: Integer;
begin
  if CurStep <> ssPostInstall then
    Exit;
  WizardForm.StatusLabel.Caption := CustomMessage('Installing');
  { přibalený Python spustí instalátor přímo (-E -s: bez PYTHONHOME / PYTHONPATH a uživatelských balíčků jiného
    Pythonu); PowerShell jen u balíčku bez Pythonu }
  Exe := ExpandConstant('{app}\tools\python\{#PyDir}\python.exe');
  if FileExists(Exe) then
    Params := '-E -s "' + ExpandConstant('{app}\installer\fea_setup.py') + '" ' + SetupArgs()
  else
  begin
    Exe := 'powershell.exe';
    Params := '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\installer\install.ps1') + '" ' + SetupArgs();
  end;
  Log('fea_setup: ' + Exe + ' ' + Params);
  if WizardSilent then
    ShowCmd := SW_HIDE
  else
    ShowCmd := SW_SHOW;
  if not Exec(Exe, Params, ExpandConstant('{app}'), ShowCmd, ewWaitUntilTerminated, Code) then
    Code := 9999;
  Log('fea_setup: exit code ' + IntToStr(Code));
  if Code <> 0 then
  begin
    InstallResult := Code;
    SuppressibleMsgBox(FmtMessage(CustomMessage('SetupFailed'), [IntToStr(Code)]), mbError, MB_OK, IDOK);
  end;
end;

function GetCustomSetupExitCode(): Integer;
begin
  Result := 0;
  if InstallResult <> 0 then
    Result := 20 + InstallResult;
end;

{ prázdné složky zdola nahoru; soubory se nikdy nemažou, do spojení (junction) a odkazů se nevstupuje.
  Vrací True, pokud se smazala i složka Dir. }
function RemoveEmptyDirs(const Dir: String): Boolean;
var
  FindRec: TFindRec;
  Empty: Boolean;
begin
  Empty := True;
  if FindFirst(AddBackslash(Dir) + '*', FindRec) then
  begin
    try
      repeat
        if (FindRec.Name <> '.') and (FindRec.Name <> '..') then
        begin
          if ((FindRec.Attributes and FA_DIRECTORY) = 0) or ((FindRec.Attributes and FA_REPARSE_POINT) <> 0) then
            Empty := False
          else if not RemoveEmptyDirs(AddBackslash(Dir) + FindRec.Name) then
            Empty := False;
        end;
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
  end;
  Result := Empty and RemoveDir(Dir);
end;

{ odinstalátor maže jen složky, které sám vytvořil – složky z rozbaleného zipu (instalace / aktualizace ze zipu) by po
  smazání souborů zůstaly prázdné; složky s uživatelskými soubory (vlastní .fea, zálohy ukázek) zůstanou }
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    RemoveEmptyDirs(ExpandConstant('{app}'));
end;
