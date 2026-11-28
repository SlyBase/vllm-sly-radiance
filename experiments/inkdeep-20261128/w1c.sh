#!/bin/bash
# Window 1c: M1 micro-sweep only (~3 min). Same restore contract as w1b.
set -u
W=/root/inkdeep-20261128; O=$W/out/M1; GW=/usr/local/sbin/gpu-window
OWNER=daidalos-inkdeep; C=vllm-opt
CAP=/sys/class/drm/card0/device/hwmon/hwmon6/power1_cap
VRAM=/sys/class/drm/card0/device/mem_info_vram_used
IMG=vllm-sly-radiance:1.15.1-rocm10.1
ACQUIRED=0
log(){ echo "$(date -u +%H:%M:%S) $*" >> "$W/progress.log"; }
restore(){
  docker rm -f "$C" >/dev/null 2>&1
  systemctl start docker-vllm7.service >/dev/null 2>&1
  systemctl start gpu-window-guard.timer >/dev/null 2>&1
  for i in $(seq 1 90); do
    h=$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health || true)
    [ "$h" = "200" ] && break
    sleep 5
  done
  [ "$ACQUIRED" = "1" ] && "$GW" release >> "$W/progress.log" 2>&1
  log "M1 RESTORE: prod health=$h cap=$(cat $CAP 2>/dev/null)"
}
trap restore EXIT
mkdir -p "$O"
PF=$W/driver.pid
if [ -f "$PF" ]; then
  old=$(cat "$PF" 2>/dev/null || echo 0)
  if [ "$old" != "$$" ] && kill -0 "$old" 2>/dev/null && grep -qa "w1" "/proc/$old/cmdline" 2>/dev/null; then
    log "ABORT: another driver alive (pid $old)"; exit 1
  fi
fi
echo $$ > "$PF"
"$GW" acquire "$OWNER" 3600
st=$("$GW" status)
echo "$st" | grep -q "\"owner\": \"$OWNER\"" || { log "ABORT: not owner: $st"; exit 1; }
ACQUIRED=1
systemctl stop gpu-window-guard.timer
systemctl stop docker-vllm7.service
docker rm -f "$C" >/dev/null 2>&1
for i in $(seq 1 36); do
  v=$(cat "$VRAM" 2>/dev/null || echo 999999999999)
  [ "$v" -le 1073741824 ] && break
  sleep 5
done
log "M1 START cap=$(cat $CAP 2>/dev/null) vram=$(cat $VRAM)"
timeout 600 docker run --rm --name "$C" --device /dev/kfd --device /dev/dri \
  --group-add video --shm-size 8g --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0 -e GPU_MAX_HW_QUEUES=1 \
  -v "$W":/work "$IMG" python3 -I /work/m1.py > "$O/m1.log" 2>&1
rc=$?
log "M1 DONE rc=$rc"
exit 0
