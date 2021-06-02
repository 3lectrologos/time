#!/usr/bin/env bash

python exp_real.py --run $1
python exp_real.py --plot 'EGFR(A)' EGFR
python exp_real.py --plot 'PDGFRA(A)' PDGFRA
python exp_real.py --plot TP53 IDH1
python exp_real.py --plot 'MDM2(A)' 'CDK4(A)'
