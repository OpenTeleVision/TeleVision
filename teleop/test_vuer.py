from vuer import Vuer, VuerSession
from vuer.schemas import DefaultScene

app = Vuer()

@app.spawn(start=True)
async def main(session: VuerSession):
    app.set @ DefaultScene(
        camera_position=[0, 1.6, 2],  # Adjust for comfortable viewing height
        vr_mode=True
    )
    while True:
        await session.sleep(0.1)

if __name__ == "__main__":
    app.run()

