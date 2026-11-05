"""Minimal async client for QueueForge's text protocol (one line per command)."""
import asyncio
import os

QF_HOST, _, QF_PORT = os.environ.get("QUEUEFORGE_ADDR", "127.0.0.1:9000").partition(":")


class QueueForge:
    def __init__(self):
        self.reader = self.writer = None
        self.lock = asyncio.Lock()          # one command in flight at a time on the shared connection

    async def command(self, line):
        """Send one line, return the reply line. Reconnects once if the connection died."""
        async with self.lock:
            for attempt in (0, 1):
                try:
                    if self.writer is None:
                        self.reader, self.writer = await asyncio.open_connection(QF_HOST, int(QF_PORT))
                    self.writer.write(line.encode() + b"\n")
                    await self.writer.drain()
                    reply = await asyncio.wait_for(self.reader.readline(), timeout=5)
                    if not reply:
                        raise ConnectionError("broker closed the connection")
                    return reply.decode().strip()
                except (OSError, ConnectionError, asyncio.TimeoutError):
                    if self.writer is not None:
                        self.writer.close()
                    self.reader = self.writer = None
                    if attempt:
                        raise

    async def submit(self, job_type, payload, priority=1):
        reply = await self.command(f"SUBMIT {priority} {job_type} {payload}")
        if not reply.startswith("OK"):
            raise RuntimeError(f"broker said: {reply}")
        return reply.split()[1]