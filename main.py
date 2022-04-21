
import uuid
from fastapi import FastAPI, File, Query, Response, UploadFile

import requests

app = FastAPI()

@app.post("/upload_image", description="file input")
async def add_known_user_voiceid(
    session_uuid: str = Query(
        ...,
        description="file input"
    ),
    file: UploadFile = File(
        ...,
        description="Main input as audio file in WAV format with at least 16KHz sample rate. "
                    "Should contain speech of only single designated user. "
                    "If you will submit audio recording here with multiple speakers then result "
                    "will be much worse than not submitting voiceid at all. "
    ),    
):
    result = None
    try:
        file_contents = await file.read()
        path_file = f'/tmp/{uuid.uuid4()}.wav'
        async with aiofiles.open(path_file, 'wb') as out_file:
            await out_file.write(file_contents)
        # process file or file_contents bytes
        result = await ControllerRequests.do_something(request=message_request_voiceid, file=file)
    except Exception as e:
        LoggingUtils.exception_log(e)
    return Response(content=result.to_json(), media_type="application/json")

