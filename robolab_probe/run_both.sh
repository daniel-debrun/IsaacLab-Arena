#!/bin/bash
# Run the apple then hammers diagnostics under one hold of the GPU lock.
cd ~/develop/arena_wt/fix_len
export GPU_LOCK_FILE=/tmp/fix_len_inner.lock
B=~/develop/IsaacLab-Arena/robolab_parity/bin/arena_local.sh
for t in apple hammers; do
  $B ~/develop/arena_wt/fix_len -w 60 -t 2400 -- robolab_probe/len_diag.py --experiment_config robolab_probe/diag_$t.yaml \
    --experiment_output_directory /workspaces/isaaclab_arena/robolab_probe/out/$t --enable_cameras > robolab_probe/out/$t.log 2>&1
  echo "exit $?" >> robolab_probe/out/$t.log
done
