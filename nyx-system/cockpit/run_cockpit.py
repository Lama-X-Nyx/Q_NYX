"""
Launch NYX Cockpit — FastAPI backend + Next.js frontend.

Usage:
  python cockpit/run_cockpit.py              # API only (port 8100)
  python cockpit/run_cockpit.py --with-ui    # API + Next.js dev server

The backend serves mock data when no NYXRuntime is connected,
so you can develop the UI independently.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
log = logging.getLogger('nyx.cockpit')


def run_api(port: int = 8100) -> None:
    import uvicorn
    from cockpit.api.server import app
    log.info('Starting NYX Cockpit API on port %d', port)
    uvicorn.run(app, host='0.0.0.0', port=port, log_level='info')


def run_ui() -> None:
    ui_dir = HERE / 'ui'
    log.info('Starting Next.js dev server from %s', ui_dir)
    subprocess.run(['npm', 'run', 'dev'], cwd=str(ui_dir))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8100)
    parser.add_argument('--with-ui', action='store_true')
    args = parser.parse_args()

    if args.with_ui:
        api_thread = threading.Thread(target=run_api, args=(args.port,), daemon=True)
        api_thread.start()
        run_ui()
    else:
        run_api(args.port)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
