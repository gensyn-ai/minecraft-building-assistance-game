#!/bin/bash

set -e
set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if ! cd Minecraft ; then
    echo Error: run $(basename $0) in the MalmoPlatorm directory.
    exit 1
fi

cp -v $SCRIPT_DIR/data/minecraft/mods/*.jar run/mods/
cp -v $SCRIPT_DIR/data/minecraft/config/*.cfg  run/config/

SKINS_DIR=$SCRIPT_DIR/data/minecraft/skins
MOD_SKINS_DIR=run/CustomSkinLoader/LocalSkin/skins
mkdir -p $MOD_SKINS_DIR

cp $SKINS_DIR/steve.png $MOD_SKINS_DIR/ppo.png
cp $SKINS_DIR/steve.png $MOD_SKINS_DIR/ppo_0.png
cp $SKINS_DIR/steve.png $MOD_SKINS_DIR/human.png
cp $SKINS_DIR/qt_robot.png $MOD_SKINS_DIR/ppo_1.png
cp $SKINS_DIR/qt_robot.png $MOD_SKINS_DIR/assistant.png
