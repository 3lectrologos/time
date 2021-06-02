#!/usr/bin/env bash

python exp_synth.py --run $1 synth2
python exp_synth.py --plot synth2
python exp_synth.py --run $1 synth5
python exp_synth.py --plot synth5
