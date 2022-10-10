#!/bin/bash

set -e
set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if ! cd Minecraft ; then
    echo Error: run $(basename $0) in the MalmoPlatorm directory.
    exit 1
fi
curl http://people.eecs.berkeley.edu/~cassidy_laidlaw/minecraft-building-assistance-game/mods/CustomSkinLoader_ForgeLegacy-14.13-SNAPSHOT-317.jar -o run/mods/CustomSkinLoader_ForgeLegacy-14.13.jar 
curl http://people.eecs.berkeley.edu/~cassidy_laidlaw/minecraft-building-assistance-game/mods/Survival-Flight-Mod-1.11.2.jar -o run/mods/Survival-Flight-Mod-1.11.2.jar

cp $SCRIPT_DIR/data/minecraft/config/zevac.survivalflight.cfg  run/config/

SKINS_DIR=$SCRIPT_DIR/data/minecraft/skins
MOD_SKINS_DIR=run/CustomSkinLoader/LocalSkin/skins
mkdir -p $MOD_SKINS_DIR

cp $SKINS_DIR/steve.png $MOD_SKINS_DIR/ppo.png
cp $SKINS_DIR/steve.png $MOD_SKINS_DIR/ppo_0.png
cp $SKINS_DIR/steve.png $MOD_SKINS_DIR/human.png
cp $SKINS_DIR/qt_robot.png $MOD_SKINS_DIR/ppo_1.png
cp $SKINS_DIR/qt_robot.png $MOD_SKINS_DIR/assistant.png
