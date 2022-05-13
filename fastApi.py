import shutil
from fastapi import FastAPI, File, Query,  UploadFile
import uvicorn

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
    return {"info": f"file '{file_name}' saved at '{file_location}'"}

if __name__ == '__main__':
    uvicorn.run(
        'fastApi:app',
        port=8001,
        host="0.0.0.0",
        debug=True,
        reload=True
    )