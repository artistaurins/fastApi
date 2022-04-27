# import uuid
# import shutil
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, Query, Response, UploadFile

# import requests

app = FastAPI()

@app.post("/upload_video")
async def add_file(
    session_id: str = Query(...),
    file: UploadFile = File(...)
    ):

# File saving in folder (don't need that, ja pareizi sapratu)
    # with open(f'{file.filename}', "wb") as buffer:
    #     shutil.copyfileobj(file.file, buffer)

    return {file.filename: "uploaded successfully!"}

# so far don't know how to get file by session_id, cause it's defined only in @post method
@app.get("/get_file")
async def file():
    # need to get file by session_id:
    return FileResponse({"filename.mp4"}) 

# pēc tam varēs implementēt OpenCV
# labāk būtu rakstīt visu OpenCV kodu tepat vai tomēr jaunā failā? (for cleaner code purposes, labāk laikam būtu jaunā failā)
