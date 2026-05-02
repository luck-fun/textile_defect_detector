from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model
import io
from PIL import Image

app = FastAPI()

model = load_model('fabric_classifier.keras')

class_names = [
    'Vertical', 'defect free', 'hole', 
    'horizontal', 'lines', 'stain'
]

def get_verdict(class_name: str) -> str:
    return f"Вердикт: Обнаружен дефект типа '{class_name}'."

@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img = img.resize((150, 150))
    
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  
    
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    
    verdict = get_verdict(predicted_class)
    
    return {
        "class_en": predicted_class,
        "verdict": verdict
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
