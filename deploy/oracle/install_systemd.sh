#!/usr/bin/env bash
set -eu

repo_dir="${STRATEGY_ARENA_HOME:-/opt/strategy-arena}"

install -m 0644 "$repo_dir/deploy/oracle/systemd/strategy-arena-compose.service" \
  /etc/systemd/system/strategy-arena-compose.service
install -m 0644 "$repo_dir/deploy/oracle/systemd/strategy-arena-backup.service" \
  /etc/systemd/system/strategy-arena-backup.service
install -m 0644 "$repo_dir/deploy/oracle/systemd/strategy-arena-backup.timer" \
  /etc/systemd/system/strategy-arena-backup.timer

systemctl daemon-reload
systemctl enable --now strategy-arena-compose.service
systemctl enable --now strategy-arena-backup.timer
systemctl status --no-pager strategy-arena-compose.service
systemctl list-timers --no-pager strategy-arena-backup.timer
