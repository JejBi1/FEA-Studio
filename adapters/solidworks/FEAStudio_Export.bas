Attribute VB_Name = "FEAStudio_Export"
' =====================================================================================================
' FEA Studio - export from SOLIDWORKS (VBA macro; module FEAStudio_Export, started by Sub main)
'
' What the macro does:
'   1. saves the active part or assembly as STEP AP214 to %TEMP%\FEAStudio\export\<name>_<yyyymmdd_hhmmss>.step
'      (the STEP export preferences are set only for the time of the export and then restored),
'   2. writes <same name>.json next to it - part metadata (part file = name in the STEP, configuration, occurrences,
'      material, colour) in the import format of FEA Studio (fea_core/rest/README.md, section "Import from CAD"),
'   3. starts FEA Studio with the STEP: "<slozka>\.venv\Scripts\pythonw.exe" -m fea_gui "<step>"
'      (backup "<folder>\FEA Studio.bat" "<step>"); FEA Studio assigns the materials from the JSON.
'
' The FEA Studio program folder is searched in this order:
'   (1) environment variable FEA_STUDIO_HOME,
'   (2) file %APPDATA%\FEAStudio\install.json, key "home" (written by the installer),
'   (3) default installation folder %LOCALAPPDATA%\Programs\FEA Studio,
'   (4) constant FEA_STUDIO_HOME_MAKRO below.
'
' The file is deliberately ASCII only: the VBA editor reads .bas in the system code page. Messages with diacritics
' are therefore written as \uXXXX and converted by the function T(). Paths and launching go through the Unicode API
' (ShellExecuteW, GetEnvironmentVariableW, FileSystemObject), so they also work with diacritics.
' No network calls. Only APIs available from SOLIDWORKS 2020.
' NOT VERIFIED in real SOLIDWORKS (not on the development PC). Installation and button: README.md next to it.
' =====================================================================================================
Option Explicit

' ---- settings ---------------------------------------------------------------------------------------
' (4) FEA Studio program folder if the macro does not find it otherwise, e.g. "D:\Programs\FEA Studio".
Private Const FEA_STUDIO_HOME_MAKRO As String = ""
' Start FEA Studio after the export (False = only save STEP + JSON and show the path).
Private Const SPUSTIT_FEA_STUDIO As Boolean = True
' Show the message also after a successful export (otherwise only on a problem or warning).
Private Const ZPRAVA_VZDY As Boolean = False
' At most this many substitute names and occurrence names ("Deska-1", "Deska-2"...) per part in the JSON.
Private Const MAX_NAZVU As Long = 24

' ---- Windows API (Unicode - paths and variables with diacritics; VBA Environ/Shell are ANSI) ---------------
Private Declare PtrSafe Function ShellExecuteW Lib "shell32.dll" (ByVal hwnd As LongPtr, _
    ByVal lpOperation As LongPtr, ByVal lpFile As LongPtr, ByVal lpParameters As LongPtr, _
    ByVal lpDirectory As LongPtr, ByVal nShowCmd As Long) As LongPtr
Private Declare PtrSafe Function GetEnvironmentVariableW Lib "kernel32" (ByVal lpName As LongPtr, _
    ByVal lpBuffer As LongPtr, ByVal nSize As Long) As Long

' ---- state of one run -------------------------------------------------------------------------------
Private swApp As SldWorks.SldWorks
Private mFso As Object
Private mParts As Object          ' Scripting.Dictionary: key "file|configuration" -> part record (Dictionary)
Private mNotes As Collection      ' notes for the JSON (for the user and for FEA Studio)
Private mWarn As Collection       ' warnings for the message after the export
Private mHidden As Long           ' hidden components written to the metadata
Private mFailed As Long           ' components whose metadata could not be read
Private mPrefsSaved As Boolean    ' original STEP preferences are saved (restored after the export)
Private mOldAP As Long, mOldGeom As Long
Private mOldFaceEdge As Boolean, mOldSplit As Boolean, mOld3D As Boolean


' ===================================================================== texts and messages

' Converts \uXXXX to a character and \n to a new line (messages with diacritics in the ASCII file).
Private Function T(ByVal s As String) As String
    Dim i As Long, c As String, res As String
    i = 1
    Do While i <= Len(s)
        c = Mid$(s, i, 1)
        If c = "\" And Mid$(s, i + 1, 1) = "u" And i + 5 <= Len(s) Then
            res = res & ChrW$(CLng("&H" & Mid$(s, i + 2, 4)))
            i = i + 6
        ElseIf c = "\" And Mid$(s, i + 1, 1) = "n" Then
            res = res & vbCrLf
            i = i + 2
        Else
            res = res & c
            i = i + 1
        End If
    Loop
    T = res
End Function

' Message to the user (SendMsgToUser2 is Unicode, VBA MsgBox is not).
Private Function Zprava(ByVal msg As String, ByVal icon As Long) As Long
    Zprava = swApp.SendMsgToUser2("FEA Studio" & vbCrLf & vbCrLf & msg, icon, swMbOk)
End Function

Private Function WarnText() As String
    Dim w As Variant, res As String
    If mWarn Is Nothing Then Exit Function
    If mWarn.Count = 0 Then Exit Function
    res = vbCrLf & vbCrLf & T("Upozorn\u011Bn\u00ED:")
    For Each w In mWarn
        res = res & vbCrLf & "- " & w
    Next
    WarnText = res
End Function

Private Function AddNote(ByVal msg As String, ByVal showUser As Boolean) As Boolean
    mNotes.Add msg
    If showUser Then mWarn.Add msg
    AddNote = True
End Function


' ===================================================================== files and environment

Private Function Fso() As Object
    If mFso Is Nothing Then Set mFso = CreateObject("Scripting.FileSystemObject")
    Set Fso = mFso
End Function

' Environment variable of the SOLIDWORKS process (Unicode).
Private Function EnvVar(ByVal nm As String) As String
    Dim buf As String, n As Long
    buf = String$(4096, vbNullChar)
    n = GetEnvironmentVariableW(StrPtr(nm), StrPtr(buf), 4096)
    If n > 0 And n < 4096 Then EnvVar = Left$(buf, n)
End Function

' User environment variable from the registry - also applies to a variable set after SOLIDWORKS was started.
Private Function UserEnvVar(ByVal nm As String) As String
    Dim sh As Object, v As String
    On Error Resume Next
    Set sh = CreateObject("WScript.Shell")
    If sh Is Nothing Then Exit Function
    v = sh.Environment("USER").Item(nm)
    If Len(v) > 0 Then v = sh.ExpandEnvironmentStrings(v)
    UserEnvVar = v
End Function

' Text file in UTF-8 (with or without BOM); without ADODB, backup via FileSystemObject.
Private Function ReadTextFile(ByVal p As String) As String
    Dim st As Object, ts As Object, txt As String
    On Error Resume Next
    Set st = CreateObject("ADODB.Stream")
    If Not st Is Nothing Then
        st.Type = 2                       ' adTypeText
        st.Charset = "utf-8"
        st.Open
        st.LoadFromFile p
        If Err.Number = 0 Then txt = st.ReadText(-1)   ' adReadAll
        st.Close
    End If
    If Err.Number <> 0 Or Len(txt) = 0 Then
        Err.Clear
        txt = ""
        Set ts = Fso().OpenTextFile(p, 1, False, 0)
        If Not ts Is Nothing Then
            txt = ts.ReadAll
            ts.Close
        End If
    End If
    ReadTextFile = txt
End Function

' Writes text that is entirely ASCII (JSON from JsonStr) - the result is valid UTF-8 without BOM.
Private Function WriteAscii(ByVal p As String, ByVal txt As String) As Boolean
    Dim ts As Object
    Set ts = Fso().CreateTextFile(p, True, False)
    ts.Write txt
    ts.Close
    WriteAscii = True
End Function

' %TEMP%\FEAStudio\export (folders are created level by level).
Private Function ExportFolder() As String
    Dim d As String
    d = EnvVar("TEMP")
    If Len(d) = 0 Then d = Fso().GetSpecialFolder(2).Path
    If Right$(d, 1) = "\" Then d = Left$(d, Len(d) - 1)
    d = d & "\FEAStudio"
    If Not Fso().FolderExists(d) Then Fso().CreateFolder d
    d = d & "\export"
    If Not Fso().FolderExists(d) Then Fso().CreateFolder d
    ExportFolder = d
End Function

' File name: without characters Windows does not allow in names (diacritics stay), at most 60 characters.
Private Function SafeName(ByVal s As String) As String
    Dim i As Long, c As String, code As Long, res As String
    For i = 1 To Len(s)
        c = Mid$(s, i, 1)
        code = AscW(c)
        If code < 0 Then code = code + 65536
        If code < 32 Or InStr(1, "\/:*?""<>| " & vbTab, c, vbBinaryCompare) > 0 Then c = "_"
        If Not (c = "_" And Right$(res, 1) = "_") Then res = res & c
    Next
    Do While Len(res) > 0 And (Left$(res, 1) = "_" Or Left$(res, 1) = ".")
        res = Mid$(res, 2)
    Loop
    Do While Len(res) > 0 And (Right$(res, 1) = "_" Or Right$(res, 1) = ".")
        res = Left$(res, Len(res) - 1)
    Loop
    If Len(res) > 60 Then res = Left$(res, 60)
    If Len(res) = 0 Then res = "model"
    SafeName = res
End Function

Private Function CleanPath(ByVal s As String) As String
    s = Trim$(s)
    If Len(s) >= 2 And Left$(s, 1) = """" And Right$(s, 1) = """" Then s = Mid$(s, 2, Len(s) - 2)
    Do While Len(s) > 3 And (Right$(s, 1) = "\" Or Right$(s, 1) = "/")
        s = Left$(s, Len(s) - 1)
    Loop
    CleanPath = s
End Function


' ===================================================================== JSON

' JSON string in ASCII: non-ASCII and control characters as \uXXXX (Czech in names passes unchanged).
Private Function JsonStr(ByVal s As String) As String
    Dim i As Long, c As String, code As Long, res As String
    For i = 1 To Len(s)
        c = Mid$(s, i, 1)
        code = AscW(c)
        If code < 0 Then code = code + 65536
        If c = """" Then
            res = res & "\"""
        ElseIf c = "\" Then
            res = res & "\\"
        ElseIf code < 32 Or code > 126 Then
            res = res & "\u" & Right$("000" & Hex$(code), 4)
        Else
            res = res & c
        End If
    Next
    JsonStr = """" & res & """"
End Function

' Array of JSON strings from the keys of Scripting.Dictionary (insertion order).
Private Function JsonList(ByVal d As Object) As String
    Dim k As Variant, res As String
    For Each k In d.Keys
        If Len(res) > 0 Then res = res & ", "
        res = res & JsonStr(CStr(k))
    Next
    JsonList = "[" & res & "]"
End Function

Private Function JsonNotes() As String
    Dim w As Variant, res As String
    For Each w In mNotes
        If Len(res) > 0 Then res = res & ", "
        res = res & JsonStr(CStr(w))
    Next
    JsonNotes = "[" & res & "]"
End Function

' Number 0-1 with a decimal point (Format/CStr would use a comma according to the Windows settings).
Private Function Num01(ByVal v As Double) As String
    Dim n As Long
    If v <= 0 Then
        Num01 = "0"
    ElseIf v >= 1 Then
        Num01 = "1"
    Else
        n = CLng(v * 10000)
        If n >= 10000 Then
            Num01 = "1"
        Else
            Num01 = "0." & Right$("0000" & CStr(n), 4)
        End If
    End If
End Function

' Colour [r, g, b] 0-1 from the SOLIDWORKS appearance property array [R, G, B, Ambient, ...]; -1 = no appearance.
Private Function ColorJson(ByVal v As Variant) As String
    Dim i0 As Long
    On Error GoTo Konec
    If Not IsArray(v) Then Exit Function
    i0 = LBound(v)
    If UBound(v) - i0 < 2 Then Exit Function
    If v(i0) < 0 Or v(i0 + 1) < 0 Or v(i0 + 2) < 0 Then Exit Function
    ColorJson = "[" & Num01(v(i0)) & ", " & Num01(v(i0 + 1)) & ", " & Num01(v(i0 + 2)) & "]"
Konec:
End Function

' Value of a string key from simple JSON (install.json), unescaped.
Private Function JsonGetString(ByVal txt As String, ByVal key As String) As String
    Dim p As Long, i As Long, c As String, res As String, pat As String
    pat = """" & key & """"
    p = InStr(1, txt, pat, vbBinaryCompare)
    Do While p > 0
        i = p + Len(pat)
        Do While i <= Len(txt)
            c = Mid$(txt, i, 1)
            If c = " " Or c = vbTab Or c = vbCr Or c = vbLf Then
                i = i + 1
            Else
                Exit Do
            End If
        Loop
        If Mid$(txt, i, 1) = ":" Then
            i = i + 1
            Do While i <= Len(txt)
                c = Mid$(txt, i, 1)
                If c = " " Or c = vbTab Or c = vbCr Or c = vbLf Then
                    i = i + 1
                Else
                    Exit Do
                End If
            Loop
            If Mid$(txt, i, 1) = """" Then
                i = i + 1
                res = ""
                Do While i <= Len(txt)
                    c = Mid$(txt, i, 1)
                    If c = """" Then
                        JsonGetString = res
                        Exit Function
                    ElseIf c = "\" Then
                        c = Mid$(txt, i + 1, 1)
                        Select Case c
                            Case "u"
                                res = res & ChrW$(CLng("&H" & Mid$(txt, i + 2, 4)))
                                i = i + 4
                            Case "n"
                                res = res & vbLf
                            Case "t"
                                res = res & vbTab
                            Case "r"
                                res = res & vbCr
                            Case "b", "f"
                            Case Else
                                res = res & c          ' \\  \"  \/
                        End Select
                        i = i + 2
                    Else
                        res = res & c
                        i = i + 1
                    End If
                Loop
            End If
        End If
        p = InStr(p + 1, txt, pat, vbBinaryCompare)
    Loop
End Function


' ===================================================================== FEA Studio folder and launch

Private Function HomeFromInstallJson() As String
    Dim ad As String, p As String
    On Error Resume Next                 ' faulty install.json = as if it did not exist
    ad = EnvVar("APPDATA")
    If Len(ad) = 0 Then Exit Function
    p = ad & "\FEAStudio\install.json"
    If Fso().FileExists(p) Then HomeFromInstallJson = JsonGetString(ReadTextFile(p), "home")
End Function

Private Function HomeValid(ByVal h As String) As Boolean
    If Len(h) = 0 Then Exit Function
    If Not Fso().FolderExists(h) Then Exit Function
    HomeValid = Fso().FileExists(h & "\.venv\Scripts\pythonw.exe") Or Fso().FileExists(h & "\FEA Studio.bat")
End Function

' Program folder: FEA_STUDIO_HOME > install.json > %LOCALAPPDATA%\Programs\FEA Studio > constant.
' tried = list of tried folders for the message.
Private Function FindHome(ByRef tried As String) As String
    Dim c(1 To 5) As String, i As Long, h As String, la As String
    c(1) = EnvVar("FEA_STUDIO_HOME")
    c(2) = UserEnvVar("FEA_STUDIO_HOME")
    c(3) = HomeFromInstallJson()
    la = EnvVar("LOCALAPPDATA")
    If Len(la) > 0 Then c(4) = la & "\Programs\FEA Studio"
    c(5) = FEA_STUDIO_HOME_MAKRO
    For i = 1 To 5
        h = CleanPath(c(i))
        If Len(h) > 0 Then
            If InStr(1, tried, h & vbCrLf, vbTextCompare) = 0 Then tried = tried & "  " & h & vbCrLf
            If HomeValid(h) Then
                FindHome = h
                Exit Function
            End If
        End If
    Next
End Function

' Launch without waiting: pythonw.exe -m fea_gui "<step>" with the program folder as working folder; backup FEA Studio.bat.
Private Function LaunchFea(ByVal home As String, ByVal stepPath As String) As Boolean
    Dim verb As String, exe As String, args As String, r As LongPtr
    verb = "open"
    exe = home & "\.venv\Scripts\pythonw.exe"
    If Fso().FileExists(exe) Then
        args = "-m fea_gui """ & stepPath & """"
        r = ShellExecuteW(0, StrPtr(verb), StrPtr(exe), StrPtr(args), StrPtr(home), 1)      ' SW_SHOWNORMAL
        If r > 32 Then
            LaunchFea = True
            Exit Function
        End If
    End If
    exe = home & "\FEA Studio.bat"
    If Fso().FileExists(exe) Then
        args = """" & stepPath & """"
        r = ShellExecuteW(0, StrPtr(verb), StrPtr(exe), StrPtr(args), StrPtr(home), 7)      ' SW_SHOWMINNOACTIVE
        LaunchFea = (r > 32)
    End If
End Function


' ===================================================================== document and STEP preferences

Private Function DocumentName(ByVal m As SldWorks.ModelDoc2) As String
    Dim p As String, tt As String
    p = m.GetPathName
    If Len(p) > 0 Then
        DocumentName = Fso().GetBaseName(p)
        Exit Function
    End If
    ' unsaved document: window title (extension according to the Explorer setting, possibly "*" for changes)
    tt = Trim$(m.GetTitle)
    If Right$(tt, 1) = "*" Then tt = Trim$(Left$(tt, Len(tt) - 1))
    Select Case LCase$(Right$(tt, 7))
        Case ".sldprt", ".sldasm"
            tt = Left$(tt, Len(tt) - 7)
    End Select
    If Len(tt) = 0 Then tt = "model"
    DocumentName = tt
End Function

Private Function ActiveConfigName(ByVal m As SldWorks.ModelDoc2) As String
    On Error Resume Next
    ActiveConfigName = m.ConfigurationManager.ActiveConfiguration.Name
End Function

Private Function SourceVersion() As String
    On Error Resume Next
    SourceVersion = swApp.RevisionNumber
End Function

' STEP export preferences are system-wide (registry) - save, set, restore after the export.
Private Function SavePrefs() As Boolean
    On Error Resume Next
    Err.Clear
    mOldAP = swApp.GetUserPreferenceIntegerValue(swStepAP)
    mOldGeom = swApp.GetUserPreferenceIntegerValue(swStepExportPreference)
    mOldFaceEdge = swApp.GetUserPreferenceToggle(swStepExportFaceEdgeProps)
    mOldSplit = swApp.GetUserPreferenceToggle(swStepExportSplitPeriodic)
    mOld3D = swApp.GetUserPreferenceToggle(swStepExport3DCurveFeatures)
    mPrefsSaved = (Err.Number = 0)
    SavePrefs = mPrefsSaved
End Function

Private Function SetPrefs() As Boolean
    If Not mPrefsSaved Then Exit Function
    On Error Resume Next
    swApp.SetUserPreferenceIntegerValue swStepAP, 214                       ' AP214 (colours); 203 = AP203
    swApp.SetUserPreferenceIntegerValue swStepExportPreference, swAcisOutputAsSolidAndSurface
    swApp.SetUserPreferenceToggle swStepExportFaceEdgeProps, True           ' face properties and colours
    swApp.SetUserPreferenceToggle swStepExportSplitPeriodic, False          ' hole = one cylindrical face
    swApp.SetUserPreferenceToggle swStepExport3DCurveFeatures, False        ' without wireframe curves
    SetPrefs = (Err.Number = 0)
End Function

Private Function RestorePrefs() As Boolean
    If Not mPrefsSaved Then Exit Function
    On Error Resume Next
    swApp.SetUserPreferenceIntegerValue swStepAP, mOldAP
    swApp.SetUserPreferenceIntegerValue swStepExportPreference, mOldGeom
    swApp.SetUserPreferenceToggle swStepExportFaceEdgeProps, mOldFaceEdge
    swApp.SetUserPreferenceToggle swStepExportSplitPeriodic, mOldSplit
    swApp.SetUserPreferenceToggle swStepExport3DCurveFeatures, mOld3D
    mPrefsSaved = False
    RestorePrefs = True
End Function

' STEP export. The format is given by the .step extension, the AP version by the swStepAP preference. SaveAs (not SaveAs3) because of
' SOLIDWORKS 2020 SP0/SP1. Selected faces/bodies would be exported alone -> the selection is cleared first.
Private Function ExportStep(ByVal m As SldWorks.ModelDoc2, ByVal stepPath As String, _
                            ByRef errs As Long, ByRef warns As Long) As Boolean
    Dim ok As Boolean
    SavePrefs
    SetPrefs
    m.ClearSelection2 True
    ok = m.Extension.SaveAs(stepPath, swSaveAsCurrentVersion, swSaveAsOptions_Silent Or swSaveAsOptions_Copy, _
                            Nothing, errs, warns)
    RestorePrefs
    ExportStep = ok
End Function

' The option "Export assembly components as separate STEP files" (SOLIDWORKS 2023+) creates
' further STEP files next to the main file - FEA Studio needs one file with all parts.
Private Function CheckSeparateFiles(ByVal folder As String, ByVal stepPath As String, ByVal t0 As Date) As Long
    Dim f As Object, ext As String, n As Long
    On Error Resume Next
    For Each f In Fso().GetFolder(folder).Files
        ext = LCase$(Fso().GetExtensionName(f.Name))
        If (ext = "step" Or ext = "stp") And StrComp(f.Path, stepPath, vbTextCompare) <> 0 Then
            If f.DateLastModified >= t0 - TimeSerial(0, 0, 2) Then n = n + 1
        End If
    Next
    If n > 0 Then
        AddNote T("SOLIDWORKS ulo\u017Eil sou\u010D\u00E1sti jako samostatn\u00E9 soubory STEP (") & n & _
                T("). Vypn\u011Bte v N\u00E1stroje > Volby > Mo\u017Enosti syst\u00E9mu > Export (form\u00E1t STEP) volbu " & _
                  "\u201EExportovat sou\u010D\u00E1sti sestavy jako samostatn\u00E9 soubory STEP\u201C a export opakujte."), True
    End If
    CheckSeparateFiles = n
End Function


' ===================================================================== part metadata

Private Function IsGeneric(ByVal s As String) As Boolean
    Dim lst As String
    lst = "|default|solid|body|solid1|body1|part|part1|component|shape|compound|teleso|" & _
          T("t\u011Bleso|v\u00FDchoz\u00ED|")
    IsGeneric = (Len(Trim$(s)) = 0) Or (InStr(1, lst, "|" & LCase$(Trim$(s)) & "|", vbBinaryCompare) > 0)
End Function

' Adds a substitute part name (without duplicates and generic names such as "Default", "Body1").
Private Function AddName(ByVal e As Object, ByVal s As String) As Boolean
    Dim d As Object
    s = Trim$(s)
    If IsGeneric(s) Then Exit Function
    If StrComp(s, e("name"), vbBinaryCompare) = 0 Then Exit Function
    Set d = e("names")
    If d.Exists(s) Or d.Count >= MAX_NAZVU Then Exit Function
    d.Add s, True
    AddName = True
End Function

' Occurrence name from the tree ("Deska-1", "Pod-1/Deska-1") - only information in the "instances" field. SOLIDWORKS does not
' write it to the STEP and FEA Studio does not pair by it: otherwise "Deska-2" as an occurrence of part Deska would be confused
' with a separate part file "Deska-2.sldprt".
Private Function AddInstance(ByVal e As Object, ByVal s As String) As Boolean
    Dim d As Object
    s = Trim$(s)
    If Len(s) = 0 Then Exit Function
    Set d = e("instances")
    If d.Exists(s) Or d.Count >= MAX_NAZVU Then Exit Function
    d.Add s, True
    AddInstance = True
End Function

' nm = part name as written by the STEP (part file without extension).
Private Function NewEntry(ByVal key As String, ByVal nm As String) As Object
    Dim e As Object
    Set e = CreateObject("Scripting.Dictionary")
    e.Add "name", nm
    e.Add "names", CreateObject("Scripting.Dictionary")
    e.Add "instances", CreateObject("Scripting.Dictionary")
    e.Add "component", ""
    e.Add "material", ""
    e.Add "database", ""
    e.Add "configuration", ""
    e.Add "color", ""
    e.Add "bodies", ""
    e.Add "count", 0
    e.Add "hidden", 0
    mParts.Add key, e
    Set NewEntry = e
End Function

' "Deska-1" -> "Deska" (substitute name when the file path is missing).
Private Function StripInstance(ByVal s As String) As String
    Dim i As Long
    i = InStrRev(s, "-")
    If i > 1 Then
        If IsNumeric(Mid$(s, i + 1)) Then s = Left$(s, i - 1)
    End If
    StripInstance = s
End Function

' Part material in the configuration; "" = no material. "" as configuration applies to a part with a single configuration.
Private Function PartMaterial(ByVal m As SldWorks.ModelDoc2, ByVal cfg As String, ByRef db As String) As String
    Dim p As SldWorks.PartDoc, s As String
    On Error Resume Next
    Set p = m
    If p Is Nothing Then Exit Function
    db = ""
    s = p.GetMaterialPropertyName2(cfg, db)
    If Len(s) = 0 And Len(cfg) > 0 Then s = p.GetMaterialPropertyName2("", db)
    PartMaterial = s
End Function

' Multi-body part where some body has a different material from the part: "Body1: material; Body2: ...".
' FEA Studio reads bodies as <part>_1, <part>_2 in an order the API does not guarantee -> information only.
Private Function BodyMaterials(ByVal m As SldWorks.ModelDoc2, ByVal cfg As String, ByVal partMat As String) As String
    Dim p As SldWorks.PartDoc, v As Variant, i As Long, b As SldWorks.Body2
    Dim s As String, db As String, res As String, differs As Boolean
    On Error Resume Next
    Set p = m
    If p Is Nothing Then Exit Function
    v = p.GetBodies2(swSolidBody, False)
    If Not IsArray(v) Then Exit Function
    If UBound(v) <= LBound(v) Then Exit Function      ' one body
    For i = LBound(v) To UBound(v)
        Set b = v(i)
        s = ""
        db = ""
        s = b.GetMaterialPropertyName(cfg, db)
        If Len(s) = 0 Then s = partMat
        If StrComp(s, partMat, vbTextCompare) <> 0 Then differs = True
        If Len(res) > 0 Then res = res & "; "
        res = res & b.Name & ": " & s
    Next
    If differs Then BodyMaterials = res
End Function

Private Function ComponentMaterial(ByVal comp As SldWorks.Component2) As String
    On Error Resume Next
    ComponentMaterial = comp.GetMaterialUserName
End Function

Private Function ComponentColor(ByVal comp As SldWorks.Component2, ByVal m As SldWorks.ModelDoc2) As String
    Dim s As String
    On Error Resume Next
    s = ColorJson(comp.GetMaterialPropertyValues2(swThisConfiguration, Nothing))
    If Len(s) = 0 And Not m Is Nothing Then s = ColorJson(m.MaterialPropertyValues)
    ComponentColor = s
End Function

' True = some parent subassembly is suppressed; hid = some is hidden.
Private Function ParentSuppressed(ByVal comp As SldWorks.Component2, ByRef hid As Boolean) As Boolean
    Dim p As SldWorks.Component2, n As Long
    On Error GoTo Konec
    Set p = comp.GetParent
    Do While Not p Is Nothing
        If p.IsSuppressed Then
            ParentSuppressed = True
            Exit Function
        End If
        If p.IsHidden(False) Then hid = True
        n = n + 1
        If n > 64 Then Exit Do
        Set p = p.GetParent
    Loop
Konec:
End Function

' Standalone part.
Private Function CollectPart(ByVal m As SldWorks.ModelDoc2, ByVal docName As String, ByVal cfg As String) As Boolean
    Dim e As Object, db As String
    On Error GoTo Vada
    Set e = NewEntry("dil|" & cfg, docName)
    e("component") = docName
    e("configuration") = cfg
    If Len(cfg) > 0 Then AddName e, docName & "_" & cfg       ' STEP: non-default configuration = "<part>_<configuration>"
    AddName e, DocumentTitle(m)
    e("material") = PartMaterial(m, cfg, db)
    e("database") = db
    e("bodies") = BodyMaterials(m, cfg, e("material"))
    e("color") = ColorJson(m.MaterialPropertyValues)
    e("count") = 1
    CollectPart = True
    Exit Function
Vada:
    mFailed = mFailed + 1
End Function

Private Function DocumentTitle(ByVal m As SldWorks.ModelDoc2) As String
    Dim tt As String
    On Error Resume Next
    tt = Trim$(m.GetTitle)
    If Right$(tt, 1) = "*" Then tt = Trim$(Left$(tt, Len(tt) - 1))
    Select Case LCase$(Right$(tt, 7))
        Case ".sldprt", ".sldasm"
            tt = Left$(tt, Len(tt) - 7)
    End Select
    DocumentTitle = tt
End Function

' One component of an assembly. The same part (file + configuration) = one record with the number of occurrences.
Private Function AddComponent(ByVal obj As Object) As Boolean
    Dim comp As SldWorks.Component2, m As SldWorks.ModelDoc2, e As Object
    Dim full As String, shortNm As String, p As String, baseNm As String, cfg As String
    Dim key As String, db As String, hid As Boolean
    On Error GoTo Vada
    Set comp = obj
    If comp Is Nothing Then Exit Function
    If comp.IsSuppressed Then Exit Function                  ' a suppressed component is not in the STEP
    If ParentSuppressed(comp, hid) Then Exit Function
    If comp.IsHidden(False) Then hid = True
    p = comp.GetPathName
    cfg = comp.ReferencedConfiguration
    full = comp.Name2                                        ' "Podsestava-1/Deska-1"
    shortNm = full
    If InStrRev(full, "/") > 0 Then shortNm = Mid$(full, InStrRev(full, "/") + 1)
    Set m = comp.GetModelDoc2
    If Not m Is Nothing Then
        If m.GetType <> swDocPART Then Exit Function         ' subassembly - its parts are listed separately
    ElseIf LCase$(Right$(p, 7)) <> ".sldprt" Then
        Exit Function
    End If
    If Len(p) > 0 Then
        baseNm = Fso().GetBaseName(p)
        key = LCase$(p) & "|" & cfg
    Else
        baseNm = StripInstance(shortNm)
        key = LCase$(baseNm) & "|" & cfg
    End If
    If mParts.Exists(key) Then
        Set e = mParts(key)
    Else
        Set e = NewEntry(key, baseNm)                        ' the STEP from SOLIDWORKS carries the part file name
        e("component") = baseNm
        e("configuration") = cfg
        If Len(cfg) > 0 Then AddName e, baseNm & "_" & cfg   ' ... for a non-default configuration "<part>_<configuration>"
        If InStr(baseNm, "^") > 1 Then AddName e, Left$(baseNm, InStr(baseNm, "^") - 1)   ' virtual "Part^Assembly"
        If Not m Is Nothing Then
            e("material") = PartMaterial(m, cfg, db)
            e("database") = db
            e("bodies") = BodyMaterials(m, cfg, e("material"))
        Else
            e("material") = ComponentMaterial(comp)
        End If
        e("color") = ComponentColor(comp, m)
    End If
    AddInstance e, shortNm                                   ' occurrences for information only (not for pairing)
    If full <> shortNm Then AddInstance e, full
    e("count") = e("count") + 1
    If hid Then
        e("hidden") = e("hidden") + 1
        mHidden = mHidden + 1
    End If
    AddComponent = True
    Exit Function
Vada:
    mFailed = mFailed + 1
End Function

Private Function CollectAssembly(ByVal m As SldWorks.ModelDoc2) As Boolean
    Dim asm As SldWorks.AssemblyDoc, v As Variant, i As Long
    On Error Resume Next
    Set asm = m
    If asm Is Nothing Then Exit Function
    ' lightweight components have no document (GetModelDoc2 = Nothing) -> no material; to be solved
    If asm.GetLightWeightComponentCount > 0 Then
        asm.ResolveAllLightWeightComponents False
        AddNote T("Odleh\u010Den\u00E9 sou\u010D\u00E1sti byly kv\u016Fli materi\u00E1l\u016Fm vy\u0159e\u0161eny."), False
    End If
    Err.Clear
    v = asm.GetComponents(False)                             ' all levels, any order
    If Not IsArray(v) Then Exit Function
    On Error GoTo 0
    For i = LBound(v) To UBound(v)
        AddComponent v(i)
    Next
    CollectAssembly = True
End Function

Private Function BuildJson(ByVal docName As String, ByVal docKind As String, ByVal cfg As String) As String
    Dim k As Variant, e As Object, s As String, items As String, nl As String, res As String
    nl = vbCrLf
    For Each k In mParts.Keys
        Set e = mParts(k)
        s = "  {""name"": " & JsonStr(e("name")) & ", ""names"": " & JsonList(e("names"))
        If Len(e("component")) > 0 Then s = s & ", ""component"": " & JsonStr(e("component"))
        If Len(e("material")) > 0 Then s = s & ", ""material"": " & JsonStr(e("material"))
        If Len(e("database")) > 0 Then s = s & ", ""material_database"": " & JsonStr(e("database"))
        If Len(e("configuration")) > 0 Then s = s & ", ""configuration"": " & JsonStr(e("configuration"))
        If Len(e("color")) > 0 Then s = s & ", ""color"": " & e("color")
        s = s & ", ""count"": " & CStr(e("count"))
        If e("hidden") > 0 Then s = s & ", ""hidden"": " & CStr(e("hidden"))
        If Len(e("bodies")) > 0 Then s = s & ", ""body_materials"": " & JsonStr(e("bodies"))
        If e("instances").Count > 0 Then s = s & ", ""instances"": " & JsonList(e("instances"))
        s = s & "}"
        If Len(items) > 0 Then items = items & "," & nl
        items = items & s
    Next
    res = "{""source"": ""SOLIDWORKS""," & nl
    res = res & " ""source_version"": " & JsonStr(SourceVersion()) & "," & nl
    res = res & " ""adapter"": ""FEAStudio_Export.bas""," & nl
    res = res & " ""document"": " & JsonStr(docName) & "," & nl
    res = res & " ""document_type"": " & JsonStr(docKind) & "," & nl
    res = res & " ""configuration"": " & JsonStr(cfg) & "," & nl
    res = res & " ""units"": ""mm""," & nl
    res = res & " ""step_protocol"": ""AP214""," & nl
    res = res & " ""exported"": " & JsonStr(Format$(Now, "yyyy\-mm\-dd hh\:nn\:ss")) & "," & nl
    res = res & " ""parts"": [" & nl & items & nl & " ]," & nl
    res = res & " ""notes"": " & JsonNotes() & nl & "}" & nl
    BuildJson = res
End Function


' ===================================================================== main macro (must be the last Sub without parameters)

Sub main()
    Dim swModel As SldWorks.ModelDoc2
    Dim docType As Long, docName As String, cfg As String, docKind As String
    Dim folder As String, baseNm As String, stepPath As String, jsonPath As String
    Dim errs As Long, warns As Long, i As Long, t0 As Date
    Dim home As String, tried As String, msg As String

    On Error GoTo Chyba
    Set swApp = Application.SldWorks
    Set mParts = CreateObject("Scripting.Dictionary")
    Set mNotes = New Collection
    Set mWarn = New Collection
    mHidden = 0
    mFailed = 0
    mPrefsSaved = False

    Set swModel = swApp.ActiveDoc
    If swModel Is Nothing Then
        Zprava T("Otev\u0159ete d\u00EDl nebo sestavu, kterou chcete poslat do FEA Studia."), swMbWarning
        Exit Sub
    End If
    docType = swModel.GetType
    If docType = swDocPART Then
        docKind = "part"
    ElseIf docType = swDocASSEMBLY Then
        docKind = "assembly"
    Else
        Zprava T("V\u00FDkres do FEA Studia poslat nelze - otev\u0159ete d\u00EDl nebo sestavu."), swMbWarning
        Exit Sub
    End If
    docName = DocumentName(swModel)
    cfg = ActiveConfigName(swModel)

    ' files: %TEMP%\FEAStudio\export\<name>_<yyyymmdd_hhmmss>.step + .json
    folder = ExportFolder()
    baseNm = SafeName(docName) & "_" & Format$(Now, "yyyymmdd_hhnnss")
    stepPath = folder & "\" & baseNm & ".step"
    i = 1
    Do While Fso().FileExists(stepPath)
        i = i + 1
        stepPath = folder & "\" & baseNm & "_" & CStr(i) & ".step"
    Loop
    jsonPath = Left$(stepPath, Len(stepPath) - 5) & ".json"

    ' 1. metadata (before the export - resolving lightweight components)
    If docType = swDocASSEMBLY Then
        CollectAssembly swModel
    Else
        CollectPart swModel, docName, cfg
    End If
    If mHidden > 0 Then
        AddNote mHidden & T(" skryt\u00FDch sou\u010D\u00E1st\u00ED: SOLIDWORKS je podle verze do STEP zap\u00ED\u0161e i nezap\u00ED\u0161e " & _
                "a FEA Studio by je zas\u00ED\u0165ovalo. D\u00EDly, kter\u00E9 do v\u00FDpo\u010Dtu nepat\u0159\u00ED, rad\u011Bji potla\u010Dte."), True
    End If
    If mFailed > 0 Then
        AddNote T("U ") & mFailed & T(" sou\u010D\u00E1st\u00ED se metadata nepoda\u0159ilo p\u0159e\u010D\u00EDst - materi\u00E1l jim p\u0159i\u0159a\u010Fte ve FEA Studiu."), True
    End If

    ' 2. STEP AP214
    t0 = Now
    If Not ExportStep(swModel, stepPath, errs, warns) Or Not Fso().FileExists(stepPath) Then
        Zprava T("Export do STEP se nepoda\u0159il (k\u00F3d chyby ") & errs & T(", varov\u00E1n\u00ED ") & warns & ")." & _
               vbCrLf & stepPath, swMbStop
        Exit Sub
    End If
    CheckSeparateFiles folder, stepPath, t0

    ' 3. metadata JSON next to the STEP (ASCII = UTF-8 without BOM)
    WriteAscii jsonPath, BuildJson(docName, docKind, cfg)

    ' 4. FEA Studio
    If Not SPUSTIT_FEA_STUDIO Then
        Zprava T("STEP a metadata jsou ulo\u017Een\u00E9:") & vbCrLf & stepPath & WarnText(), swMbInformation
        Exit Sub
    End If
    home = FindHome(tried)
    If Len(home) = 0 Then
        msg = T("STEP a metadata jsou ulo\u017Een\u00E9, ale slo\u017Eku FEA Studia se nepoda\u0159ilo naj\u00EDt.\n\n" & _
                "Nastavte prom\u011Bnnou prost\u0159ed\u00ED FEA_STUDIO_HOME na slo\u017Eku programu (a restartujte SOLIDWORKS), " & _
                "nebo ji zapi\u0161te do konstanty FEA_STUDIO_HOME_MAKRO na za\u010D\u00E1tku makra. " & _
                "Soubor STEP m\u016F\u017Eete tak\u00E9 p\u0159et\u00E1hnout na okno FEA Studia.\n\nSTEP:\n") & stepPath
        If Len(tried) > 0 Then msg = msg & vbCrLf & vbCrLf & T("Zkou\u0161en\u00E9 slo\u017Eky:") & vbCrLf & tried
        Zprava msg & WarnText(), swMbWarning
        Exit Sub
    End If
    If Not LaunchFea(home, stepPath) Then
        Zprava T("FEA Studio se nepoda\u0159ilo spustit ze slo\u017Eky:") & vbCrLf & home & vbCrLf & vbCrLf & _
               T("STEP a metadata jsou ulo\u017Een\u00E9:") & vbCrLf & stepPath & WarnText(), swMbWarning
        Exit Sub
    End If
    msg = WarnText()
    If Len(msg) > 0 Or ZPRAVA_VZDY Then
        Zprava T("Model se otev\u00EDr\u00E1 ve FEA Studiu:") & vbCrLf & stepPath & msg, swMbInformation
    End If
    Exit Sub

Chyba:
    msg = Err.Description
    If mPrefsSaved Then RestorePrefs
    Zprava T("Export pro FEA Studio selhal: ") & msg, swMbStop
End Sub
