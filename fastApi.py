import os

import shutil
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI()

@app.post("/upload_file")
async def upload_file(
    session_id: str = Query(...),
    file: UploadFile = File(...)
):
    try:
        extension = file.filename.split(".")[1]
        file_name = session_id
        file_location = f"unprocessedFiles/{file_name}.{extension}"

        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        return {"info": f"file '{file_name}' saved at '{file_location}'"}
    except:
        return {"Error" : "File is not saved"}

@app.get("/processedFiles/{session_id}")
async def get_results(
    session_id: str = Query(...)
):
    file_location = f"processedFiles/{session_id}.json"
    if os.path.exists(file_location):
       return FileResponse(file_location) 
    return {"Error" : "File not found!"}

if __name__ == '__main__':
    uvicorn.run(
        'fastApi:app',
        port=8001,
        host="0.0.0.0",
        debug=True,
        reload=True
    )
