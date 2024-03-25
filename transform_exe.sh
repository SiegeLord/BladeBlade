#!/bin/bash
set -e
cp target/release/BladeBlade.exe .
magick convert raw/icon_*.png -compress zip raw/icon.ico
winpty ../ResourceHacker/ResourceHacker.exe -open BladeBlade.exe -save BladeBlade.exe -action addoverwrite -res raw/icon.ico -mask ICONGROUP,MAINICON,