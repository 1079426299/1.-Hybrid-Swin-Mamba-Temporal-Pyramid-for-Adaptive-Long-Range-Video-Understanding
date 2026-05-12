#!/usr/bin/env bash
set -e
python eval.py --config configs/smtpn_tiny_dummy.yaml --checkpoint outputs/smoke/checkpoint_last.pt --batch-size 2
