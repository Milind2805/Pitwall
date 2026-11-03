import asyncio, websockets

async def main():
    async with websockets.connect("ws://localhost:8000/ws", max_size=None) as ws:
        await ws.recv()                 # take the snapshot
        print("connected, now ignoring the stream for 30 s ...")
        await asyncio.sleep(30)         # not reading: the server-side queue backs up
        print("reading again")
        for _ in range(5):
            m = await ws.recv()
            print(m[:80])

asyncio.run(main())