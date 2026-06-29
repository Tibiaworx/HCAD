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

  ; Visual C++ runtime the Manifold kernel needs — silent, only installs if missing.
  SetOutPath "$TEMP"
  File "vc_redist.x64.exe"
  DetailPrint "Ensuring the Visual C++ runtime is installed..."
  ExecWait '"$TEMP\vc_redist.x64.exe" /install /quiet /norestart'
  Delete "$TEMP\vc_redist.x64.exe"
  SetOutPath "$INSTDIR"

  ; Shortcuts
  CreateDirectory "$SMPROGRAMS\HCAD"
  CreateShortcut "$SMPROGRAMS\HCAD\HCAD.lnk" "$INSTDIR\HCAD.exe"
  CreateShortcut "$SMPROGRAMS\HCAD\Uninstall HCAD.lnk" "$INSTDIR\Uninstall.exe"
  CreateShortcut "$DESKTOP\HCAD.lnk" "$INSTDIR\HCAD.exe"

  ; Registry: install dir + Add/Remove Programs entry.
  WriteRegStr HKLM "Software\HCAD" "InstallDir" "$INSTDIR"
  !define UNINST "Software\Microsoft\Windows\CurrentVersion\Uninstall\HCAD"
  WriteRegStr HKLM "${UNINST}" "DisplayName" "HCAD"
  WriteRegStr HKLM "${UNINST}" "DisplayVersion" "1.0.0"
  WriteRegStr HKLM "${UNINST}" "Publisher" "HCAD"
  WriteRegStr HKLM "${UNINST}" "DisplayIcon" "$INSTDIR\HCAD.exe"
  WriteRegStr HKLM "${UNINST}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr HKLM "${UNINST}" "InstallLocation" "$INSTDIR"
  WriteRegDWORD HKLM "${UNINST}" "NoModify" 1
  WriteRegDWORD HKLM "${UNINST}" "NoRepair" 1

  WriteUninstaller "$INSTDIR\Uninstall.exe"
SectionEnd

; ---- Uninstall ----
Section "Uninstall"
  Delete "$INSTDIR\HCAD.exe"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"

  Delete "$SMPROGRAMS\HCAD\HCAD.lnk"
  Delete "$SMPROGRAMS\HCAD\Uninstall HCAD.lnk"
  RMDir "$SMPROGRAMS\HCAD"
  Delete "$DESKTOP\HCAD.lnk"

  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\HCAD"
  DeleteRegKey HKLM "Software\HCAD"
SectionEnd
