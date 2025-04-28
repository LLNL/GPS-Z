#!/bin/bash

shopt -s globstar

sed "s|PWD|${PWD}|" test_inputs/project_base.json > test_inputs/project.json
cd ..

d=test_inputs

function run() {
  label=$1
  rm -rf test/${d}/GPS test/${d}/detailed/raw
  echo "Testing ${label}..."
  time python3 gps_headless.py test/${d}/project.json >& test/${label}.log
  mv test/${d}/GPS/**/chem.cti test/${label}.cti
}

unset GPS_CANTERA
unset GPS_SERIAL
unset GPS_CANTERA_CV
#Zerork+Pool
run "zerork_pool" 

unset GPS_CANTERA
export GPS_SERIAL=T
unset GPS_CANTERA_CV
#Zerork
run "zerork"

export GPS_CANTERA=T
unset GPS_SERIAL
unset GPS_CANTERA_CV
#Cantera+Pool
run "cantera_pool"

export GPS_CANTERA=T
export GPS_SERIAL=T
unset GPS_CANTERA_CV
#Cantera
run "cantera"

export GPS_CANTERA=T
export GPS_SERIAL=T
export GPS_CANTERA_CV=T
#Cantera+Serial+CV
run "cantera_cv"

