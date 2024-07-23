from fastapi import FastAPI, File, UploadFile
from tensorflow.keras import utils
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import io
import uvicorn


cnn = load_model('brain_tumor.h5')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_headers = ['*'],
    allow_credentials = True,
    allow_methods = ['*'],
    allow_origins = ['*']
)

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image_data = io.BytesIO(image_data)
    image_data = utils.load_img(image_data, target_size = (64, 64))
    image_data = utils.img_to_array(image_data)
    image_data = np.expand_dims(image_data, axis = 0)
    result = cnn.predict(image_data)
    prediction = np.argmax(result, axis = 1)[0]
    if prediction == 0:
        return 'Your brain has Glioma Tumor'
    elif prediction == 1:
        return 'Your brain has Meningioma tumor'
    elif prediction == 2:
        return 'Your brain has no tumor'
    else:
        return 'Your brain has Pituitary Tumor'
    
if __name__ == '__main__':
    uvicorn.run(app)