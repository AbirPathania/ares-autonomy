#!/bin/bash
# Run one section 10 gate end to end, inside the container.
#
#   tools/gate.sh <gate> <world> [model] [--drive]
#
#   tools/gate.sh 1 flat
#   tools/gate.sh 2 step_field ares_g --drive
#   tools/gate.sh 3 trench_field ares_g --drive
#   tools/gate.sh 3 trench_field ares_g_mast_050 --drive
#   tools/gate.sh 4 snow_scree ares_g --drive
#
# Kills everything first. Orphaned nodes from a previous run keep publishing
# and keep appending to the log, and a gate scored against a stale stereo node
# is worse than no gate.

set -u
GATE=${1:?gate number}
WORLD=${2:?world name}
MODEL=${3:-ares_g}
DRIVE=${4:-}

WS=/workspaces/ares-autonomy

cleanup() {
  for name in "gz sim" stereo_node traversability_node segmentation_node \
              detection_node parameter_bridge robot_state_publisher \
              run_gate.py ros2; do
    pkill -9 -f "$name" 2>/dev/null || true
  done
}

cleanup
sleep 3

cd "$WS" || exit 1
# shellcheck disable=SC1091
. /opt/ros/jazzy/setup.sh
# shellcheck disable=SC1091
. install/setup.sh

# Software rendering, the older engine, and the world's own textures.
export LIBGL_ALWAYS_SOFTWARE=1
export GZ_SIM_RESOURCE_PATH="$WS/ares_sim"
export PYTHONPATH="${PYTHONPATH:-}:$WS"

echo "gate $GATE, world $WORLD, model $MODEL ${DRIVE}"

ros2 launch ares_bringup sim.launch.py \
  "world:=$WORLD" "model:=$MODEL" rviz:=false > /tmp/launch.log 2>&1 &
sleep 40

ros2 launch ares_bringup perception.launch.py > /tmp/perception.log 2>&1 &
sleep 35

python3 tools/run_gate.py --gate "$GATE" --model "$MODEL" --duration 70 $DRIVE
RC=$?

echo "--- stereo"
grep -a 'paired' /tmp/perception.log | tail -1
echo "--- governor"
grep -a -E 'km/h|stopping:' /tmp/perception.log | tail -3
echo "--- errors"
grep -a -iE 'error|traceback' /tmp/launch.log /tmp/perception.log | head -5

cleanup
exit $RC
