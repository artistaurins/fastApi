import shutil
import uuid
import aiofiles
from fastapi import FastAPI, File,Query,  UploadFile

from testing import head_pose

app = FastAPI()

# File uploading
@app.post("/upload_file")
async def upload_file(
    session_id: str = Query(...),
    file: UploadFile = File(...)
):
    extension = file.filename.split(".")[1]
    file_name = session_id
    file_location = f"unprocessedFiles/{file_name}.{extension}"

    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)    

    result = None
    try:
        file_contents = await file.read()
        path_file = f'/unprocessedFiles/{uuid.uuid4()}.wav'
        async with aiofiles.open(path_file, 'wb') as out_file:
            await out_file.write(file_contents)
        # process file or file_contents bytes
        head_pose(file_name)

    #     result = await ControllerRequests.do_something(request=message_request_voiceid, file=file)
    except Exception as e:
        # LoggingUtils.exception_log(e)
    # return Response(content=result.to_json(), media_type="application/json")
        return {"info": f"file '{file_name}' saved at '{file_location}'"}

