"""parent_watchdog.py — Detect rack process death and self-terminate.

This module exists in IDENTICAL FORM in every PylonRack Python slot
(pylonrack-llama, pylonrack-calibrate, etc). If you modify it here, copy
the changes to the other slots. It is small (~40 lines) on purpose:
duplication is cheaper than coordinating a shared package across separate
git repositories.

PROBLEM:
The rack process (PylonRack.app) launches slot processes as children.
If the rack crashes hard (SIGKILL, segfault, force-quit, abrupt power
loss), it does NOT get a chance to terminate its children. macOS then
re-parents the orphaned children to launchd (PID 1), where they continue
running indefinitely, holding ports open, consuming RAM. The next time
the user launches the rack, slot startup fails because the port is busy.

SOLUTION:
At startup we record the original parent PID. A background task polls
every 2 seconds; if our parent PID changes to 1 (init/launchd), we know
the rack died and we self-terminate gracefully via SIGTERM.

WHY NOT PR_SET_PDEATHSIG?
That's a Linux-specific prctl. macOS has no direct equivalent. The
polling approach is portable and adds negligible CPU (one getppid()
syscall every 2 seconds = effectively free).
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal

log = logging.getLogger("pylonrack.watchdog")


async def watch_parent(check_interval: float = 2.0) -> None:
    """Self-terminate if the parent process dies.

    Call this as a long-running asyncio task from the slot's main():

        asyncio.create_task(watch_parent())

    Returns only if the process is being shut down (never returns normally
    while the parent is alive).
    """
    original_ppid = os.getppid()

    if original_ppid == 1:
        # We were already orphaned at startup (launched directly, not by rack).
        # Nothing useful for the watchdog to do — silently no-op.
        log.warning("Already orphaned at startup (ppid=1); watchdog disabled")
        return

    log.info("Parent watchdog active — original ppid=%d", original_ppid)

    while True:
        await asyncio.sleep(check_interval)
        current_ppid = os.getppid()
        if current_ppid == 1 or current_ppid != original_ppid:
            log.warning(
                "Parent process died (ppid changed from %d to %d) — self-terminating",
                original_ppid, current_ppid,
            )
            # SIGTERM ourselves so cleanup handlers run (close sockets,
            # flush logs). If we're still alive 3 seconds later, SIGKILL
            # via a second task as belt-and-suspenders.
            os.kill(os.getpid(), signal.SIGTERM)
            await asyncio.sleep(3)
            os.kill(os.getpid(), signal.SIGKILL)
            return
