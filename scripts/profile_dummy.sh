#!/usr/bin/env bash
set -e
python tools/profile_runtime.py --config configs/smtpn_tiny_dummy.yaml --batch-size 1 --frames 16 --height 112 --width 112
