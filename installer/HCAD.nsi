; HCAD installer (NSIS / Modern UI 2)
; Build with:  "C:\Program Files (x86)\NSIS\makensis.exe" HCAD.nsi
; Produces HCAD-Setup.exe in this folder.

!include "MUI2.nsh"

Name "HCAD"
OutFile "HCAD-Setup.exe"
Unicode True
InstallDir "$PROGRAMFILES64\HCAD"
InstallDirRegKey HKLM "Software\HCAD" "InstallDir"
RequestExecutionLevel admin   ; needed to write to Program Files
BrandingText "HCAD"

; Branding: the setup/uninstall exe icons and the wizard header use the HCAD logo.
Icon "hcad.ico"
UninstallIcon "hcad.ico"

!define MUI_ICON "hcad.ico"
!define MUI_UNICON "hcad.ico"
!define MUI_ABORTWARNING
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_RUN "$INSTDIR\HCAD.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Launch HCAD"
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

; ---- Install ----
Section "HCAD (required)" SecMain
  SectionIn RO
  SetOutPath "$INSTDIR"
  File /oname=HCAD.exe "..\target\release\hcad.exe"
  File "hcad.ico"

  ; Visual C++ runtime the Manifold kernel needs — silent, only installs if missing.
  SetOutPath "$TEMP"
  File "vc_redist.x64.exe"
  DetailPrint "Ensuring the Visual C++ runtime is installed..."
  ExecWait '"$TEMP\vc_redist.x64.exe" /install /quiet /norestart'
  Delete "$TEMP\vc_redist.x64.exe"
  SetOutPath "$INSTDIR"

  ; Shortcuts (icon from the installed .ico so they show the logo even without the exe resource).
  CreateDirectory "$SMPROGRAMS\HCAD"
  CreateShortcut "$SMPROGRAMS\HCAD\HCAD.lnk" "$INSTDIR\HCAD.exe" "" "$INSTDIR\hcad.ico"
  CreateShortcut "$SMPROGRAMS\HCAD\Uninstall HCAD.lnk" "$INSTDIR\Uninstall.exe"
  CreateShortcut "$DESKTOP\HCAD.lnk" "$INSTDIR\HCAD.exe" "" "$INSTDIR\hcad.ico"

  ; Registry: install dir + Add/Remove Programs entry.
  WriteRegStr HKLM "Software\HCAD" "InstallDir" "$INSTDIR"
  !define UNINST "Software\Microsoft\Windows\CurrentVersion\Uninstall\HCAD"
  WriteRegStr HKLM "${UNINST}" "DisplayName" "HCAD"
  WriteRegStr HKLM "${UNINST}" "DisplayVersion" "0.17.0"
  WriteRegStr HKLM "${UNINST}" "Publisher" "HCAD"
  WriteRegStr HKLM "${UNINST}" "DisplayIcon" "$INSTDIR\hcad.ico"
  WriteRegStr HKLM "${UNINST}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr HKLM "${UNINST}" "InstallLocation" "$INSTDIR"
  WriteRegDWORD HKLM "${UNINST}" "NoModify" 1
  WriteRegDWORD HKLM "${UNINST}" "NoRepair" 1

  ; .hcad file association: double-click opens the part in HCAD, files show the gear icon.
  WriteRegStr HKLM "Software\Classes\.hcad" "" "HCAD.Part"
  WriteRegStr HKLM "Software\Classes\HCAD.Part" "" "HCAD Part"
  WriteRegStr HKLM "Software\Classes\HCAD.Part\DefaultIcon" "" "$INSTDIR\hcad.ico"
  WriteRegStr HKLM "Software\Classes\HCAD.Part\shell\open\command" "" '"$INSTDIR\HCAD.exe" "%1"'
  ; Tell Explorer the associations changed (SHCNE_ASSOCCHANGED, SHCNF_IDLIST).
  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, p 0, p 0)'

  WriteUninstaller "$INSTDIR\Uninstall.exe"
SectionEnd

; ---- Uninstall ----
Section "Uninstall"
  Delete "$INSTDIR\HCAD.exe"
  Delete "$INSTDIR\hcad.ico"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"

  Delete "$SMPROGRAMS\HCAD\HCAD.lnk"
  Delete "$SMPROGRAMS\HCAD\Uninstall HCAD.lnk"
  RMDir "$SMPROGRAMS\HCAD"
  Delete "$DESKTOP\HCAD.lnk"

  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\HCAD"
  DeleteRegKey HKLM "Software\HCAD"
  DeleteRegKey HKLM "Software\Classes\HCAD.Part"
  DeleteRegKey HKLM "Software\Classes\.hcad"
  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, p 0, p 0)'
SectionEnd
