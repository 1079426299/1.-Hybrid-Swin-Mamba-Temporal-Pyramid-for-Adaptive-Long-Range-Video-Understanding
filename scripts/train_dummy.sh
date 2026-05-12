#!/usr/bin/env bash
set -e
python train.py --config configs/smtpn_tiny_dummy.yaml --epochs 1 --batch-size 2 --num-workers 0 --output outputs/smoke
