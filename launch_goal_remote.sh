#!/usr/bin/env bash
set -euo pipefail
mkdir -p /root/whisper-pularr/logs
cd /root/whisper-pularr
nohup bash remote/run_goal_h100_fast.sh \
  /root/whisper-pularr \
  google/WaxalNLP \
  ful_asr \
  openai/whisper-small \
  openai/whisper-large-v3 \
  '' \
  ngia/ASR_pulaar \
  default \
  > /root/whisper-pularr/logs/goal_h100_fast.log 2>&1 < /dev/null &
echo $!
