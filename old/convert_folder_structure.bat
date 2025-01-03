@echo off
setlocal enabledelayedexpansion

:: Quell- und Zielverzeichnisse anpassen
set "source_dir=C:\Users\Lukas\AppData\LocalLow\DefaultCompany\Fuwa_HDRP\singleBuildings_ResnetSam_2"
set "target_dir=C:\Users\Lukas\AppData\LocalLow\DefaultCompany\Fuwa_HDRP\Recordings_Single"

:: Zielverzeichnis erstellen, falls nicht vorhanden
if not exist "%target_dir%" (
    mkdir "%target_dir%"
)

:: Liste der Gebäude
set buildings=A-Building B-Building C-Building E-Building F-Building G-Building H-Building I-Building M-Building N-Building L-Building O-Building R-Building Z-Building

:: Iteration über alle Gebäude
for %%b in (%buildings%) do (
    echo Bearbeite Gebäude: %%b

    :: Pfad zum aktuellen Gebäude
    set "building_path=%source_dir%\%%b"

    if exist "!building_path!" (
        :: Iteration über die sequence.*-Ordner
        for /d %%s in ("!building_path!\sequence.*") do (
            :: Sequenznummer extrahieren
            for %%a in (%%s) do (
                set "sequence_number=%%~nxa"
                set "sequence_number=!sequence_number:sequence.=!"
            )

            :: Quelldatei definieren
            set "source_file=%%s\step0.camera.png"

            :: Zieldatei definieren
            set "target_file=%target_dir%\Fuwa_%%b_!sequence_number!.png"

            :: Datei kopieren
            if exist "!source_file!" (
                echo Kopiere !source_file! nach !target_file!
                xcopy "!source_file!" "!target_file!" /Y
                echo Kopiert: !source_file! -> !target_file!
            ) else (
                echo WARNUNG: Datei fehlt: !source_file!
            )
        )
    ) else (
        echo WARNUNG: Gebäudeordner fehlt: !building_path!
    )
)

echo Konvertierung abgeschlossen!
pause
