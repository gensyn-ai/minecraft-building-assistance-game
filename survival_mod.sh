#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd Minecraft
curl https://www.curseforge.com/minecraft/mc-mods/survival-flight/download/3388089/file -o .minecraft/mods/Survival-Flight.jar