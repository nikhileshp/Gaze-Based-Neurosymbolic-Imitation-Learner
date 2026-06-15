#!/usr/bin/env bash
# remote_gui.sh — Persistent virtual-display + VNC for the pygame GUIs.
#
# The VNC server (Xvfb + x11vnc) stays UP across GUI restarts and crashes, so
# you connect your viewer ONCE and relaunch GUIs freely. (The previous version
# tied x11vnc to the GUI process, so a GUI that exited instantly — e.g. "no
# checkpoint found" — tore the VNC down before you could connect.)
#
# Why a virtual display: the server's real desktop :0 is Xwayland-rootless,
# which x11vnc cannot screen-grab (BadMatch on X_GetImage). Xvfb grabs cleanly.
#
# Subcommands:
#   serve            start (or reuse) Xvfb + x11vnc, then exit (daemons persist)
#   stop             tear down Xvfb + x11vnc
#   status           show what's running
#   <command...>     ensure server is up, then run the command on the display in
#                    the foreground; the VNC stays up after it exits
#
# Typical use:
#   ./scripts/play/remote_gui.sh serve                      # bring VNC up once
#   # laptop: ssh -L 5900:localhost:5900 nick@<server> ; vncviewer localhost:5900
#   ./scripts/play/remote_gui.sh python scripts/play/play_gui.py -g seaquest -a <ckpt>
#
# Env knobs: VNC_DISPLAY (:100)  VNC_PORT (5900)  VNC_GEOM (1280x720x24)
# Optional password: create ~/.vnc/passwd via `x11vnc -storepasswd`; else the
# port is unauthenticated but bound to 127.0.0.1 (reachable only via SSH tunnel).
set -euo pipefail

DISP="${VNC_DISPLAY:-:100}"
PORT="${VNC_PORT:-5900}"
GEOM="${VNC_GEOM:-1280x720x24}"
DNUM="${DISP#:}"
XVFB_LOG="/tmp/grail-xvfb${DISP}.log"
VNC_LOG="/tmp/grail-x11vnc-${PORT}.log"

_xvfb_up() { [ -S "/tmp/.X11-unix/X${DNUM}" ]; }
_vnc_up()  { ss -ltn 2>/dev/null | grep -q "127.0.0.1:${PORT} "; }

start_server() {
    if _xvfb_up; then
        echo "[remote_gui] reusing Xvfb ${DISP}"
    else
        # -ac so any of your shells can use DISPLAY=${DISP} without juggling an
        # Xauthority; -nolisten tcp keeps it off the network.
        setsid bash -c "exec Xvfb ${DISP} -screen 0 ${GEOM} -ac -nolisten tcp" \
            >"$XVFB_LOG" 2>&1 </dev/null &
        for _ in $(seq 1 50); do _xvfb_up && break; sleep 0.1 || true; done
        _xvfb_up || { echo "[remote_gui] ERROR: Xvfb ${DISP} failed; see $XVFB_LOG" >&2; exit 1; }
        echo "[remote_gui] started Xvfb ${DISP} (${GEOM})"
    fi

    if _vnc_up; then
        echo "[remote_gui] reusing x11vnc on 127.0.0.1:${PORT}"
    else
        local auth="-nopw"
        [ -f "$HOME/.vnc/passwd" ] && auth="-rfbauth $HOME/.vnc/passwd"
        setsid bash -c "exec x11vnc -display ${DISP} ${auth} -localhost -rfbport ${PORT} -shared -forever -noxdamage -quiet -o '${VNC_LOG}'" \
            >/dev/null 2>&1 </dev/null &
        for _ in $(seq 1 50); do _vnc_up && break; sleep 0.1 || true; done
        _vnc_up || { echo "[remote_gui] ERROR: x11vnc not listening on ${PORT}; see $VNC_LOG" >&2; exit 1; }
        echo "[remote_gui] started x11vnc on 127.0.0.1:${PORT}"
    fi
    echo "[remote_gui] laptop: ssh -L ${PORT}:localhost:${PORT} ${USER}@<server>  then VNC viewer -> localhost:${PORT}"
}

stop_server() {
    pkill -f "x11vnc -display ${DISP} " 2>/dev/null && echo "[remote_gui] stopped x11vnc" || true
    pkill -f "Xvfb ${DISP} "          2>/dev/null && echo "[remote_gui] stopped Xvfb ${DISP}" || true
}

show_status() {
    _xvfb_up && echo "Xvfb ${DISP}:  UP"        || echo "Xvfb ${DISP}:  down"
    _vnc_up  && echo "x11vnc ${PORT}: LISTENING" || echo "x11vnc ${PORT}: down"
}

case "${1:-}" in
    serve)  start_server ;;
    stop)   stop_server ;;
    status) show_status ;;
    "")     echo "usage: $0 {serve|stop|status| <command...>}" >&2; exit 2 ;;
    *)
        start_server
        echo "[remote_gui] running on ${DISP}: $*"
        DISPLAY="${DISP}" exec "$@"
        ;;
esac
