#!/usr/bin/env bash
# Entry point of the single-container demo: uvicorn on loopback, nginx in front.
#
# nginx starts only once the API answers /health, so the platform keeps showing
# its "starting" page until the console actually works instead of serving 502s
# while the embedding model loads. After that, the first process to exit takes
# the container down with it, and the platform restarts it: a dead API behind a
# live nginx would look up while answering nothing.
set -euo pipefail

uvicorn aegis.interface.api.main:app \
    --host 127.0.0.1 --port 8000 --workers 1 --no-access-log &
api_pid=$!

trap 'kill -TERM "$api_pid" ${nginx_pid:-} 2>/dev/null || true' TERM INT

for _ in $(seq 1 300); do
    if ! kill -0 "$api_pid" 2>/dev/null; then
        status=0
        wait "$api_pid" || status=$?
        exit "$status"
    fi
    if curl -fsS -o /dev/null http://127.0.0.1:8000/health; then
        break
    fi
    sleep 1
done

nginx -e /dev/stderr -g 'daemon off;' &
nginx_pid=$!

status=0
wait -n "$api_pid" "$nginx_pid" || status=$?
kill -TERM "$api_pid" "$nginx_pid" 2>/dev/null || true
exit "$status"
