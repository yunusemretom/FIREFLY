#!/usr/bin/env bash
# Simulatoru (Wine) acik degilse baslatir ve TCP :12345 hazir olana kadar bekler.
#   scripts/sim_up.sh            # gerekirse baslat + bekle
#   SIM_DIR=... scripts/sim_up.sh
set -u
SIM_DIR="${SIM_DIR:-$HOME/Desktop/Drones of War Teknofest/DronesOfWar/Binaries/Win64}"
EXE="DronesOfWar-Win64-Shipping.exe"
LOG="${SIM_LOG:-/tmp/dronesofwar_sim.log}"
TIMEOUT="${SIM_TIMEOUT:-120}"

port_up() { ss -ltn 2>/dev/null | grep -q ':12345 '; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Telemetri gercekten akiyor mu? (cokmus surec portu tutmaya devam edebilir)
telemetry_alive() {
  timeout 6 python3 - "$ROOT" <<'PY' >/dev/null 2>&1
import sys, time
sys.path.insert(0, sys.argv[1])
import drone_sdk as d
if not d.connect():
    sys.exit(1)
# Hedef (Talon) surekli ucar: yonelimi ~100 Hz degisir. Degismiyorsa oyun donmus.
time.sleep(0.5); a = d.get_target_rotation(); time.sleep(0.6); b = d.get_target_rotation()
d.disconnect()
sys.exit(0 if a != b else 1)
PY
}

if port_up && telemetry_alive; then
  echo "simulator zaten acik ve telemetri akiyor"
  exit 0
fi

# Asili/cokmus surec varsa temizle (crash reporter dahil)
pkill -f "$EXE" 2>/dev/null
pkill -f CrashReportClient 2>/dev/null
sleep 3

echo "simulator baslatiliyor ($SIM_DIR)"
cd "$SIM_DIR" || { echo "dizin yok: $SIM_DIR"; exit 1; }
DISPLAY="${DISPLAY:-:1}" nohup wine "$EXE" >"$LOG" 2>&1 &

for ((i = 0; i < TIMEOUT; i++)); do
  if port_up; then
    echo "hazir (${i}s) — TCP :12345 dinliyor"
    exit 0
  fi
  sleep 1
done
echo "zaman asimi: ${TIMEOUT}s icinde :12345 acilmadi (log: $LOG)"
exit 1
