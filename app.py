from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import io, os
from PIL import Image
import numpy as np
import gdown
from skimage.color import rgb2lab, lab2rgb, gray2rgb
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
MODEL_DIR = 'models'
MODEL_FILENAME = 'PRelu_Colorization_Model.h5'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Google Drive file ID ของโมเดล (.h5) ที่คุณแชร์ไว้
GDRIVE_FILE_ID = '1aBcD3EfGHIjkLmNOpQrStUvWXyZ12345'
GDRIVE_URL = f'https://drive.google.com/file/d/1q5Iz4hIDuq0Yv9ikpBW1_OdieiqIy4S7/view?usp=sharing'

# สร้างโฟลเดอร์เก็บโมเดล (ถ้ายังไม่มี)
os.makedirs(MODEL_DIR, exist_ok=True)

# ดาวน์โหลดโมเดลจาก Google Drive ถ้าไฟล์ยังไม่มีในเครื่อง
if not os.path.isfile(MODEL_PATH):
    print(f"Downloading model to {MODEL_PATH} ...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    print("Download complete.")

# --- โหลด InceptionResNetV2 สำหรับสร้าง embedding ---
inception = InceptionResNetV2(weights='imagenet', include_top=True)

def create_inception_embedding(grayscaled_rgb):
    # grayscaled_rgb: list of (H,W,3) numpy arrays scaled [0,1]
    x = np.array([
        img_to_array(
            Image.fromarray((img * 255).astype(np.uint8))
            .resize((299, 299))
        ) for img in grayscaled_rgb
    ])
    x = preprocess_input(x)
    return inception.predict(x)

# --- โหลดโมเดล colorization ของคุณ (จากไฟล์ที่ดาวน์โหลดมา) ---
color_model = load_model(MODEL_PATH, compile=False)
color_model.compile(optimizer='adam', loss='mean_squared_error')

def colorize_image(pil_gray):
    # แปลงเป็นขาวดำขนาด 256x256
    gray = pil_gray.convert('L').resize((256, 256))
    gray_arr = np.array(gray)

    # เตรียม embedding input
    rgb_input = gray2rgb(gray_arr.astype(np.uint8) / 255.0)
    embed = create_inception_embedding([rgb_input])

    # เตรียม L-channel input
    lab = rgb2lab([rgb_input])
    L = lab[:, :, :, 0]
    L = L.reshape(L.shape + (1,))

    # ทำนาย a,b channels
    ab = color_model.predict([L, embed])
    ab = ab * 128  # scale back

    # รวม L+a+b แล้วแปลงเป็น RGB
    lab_out = np.zeros((256, 256, 3))
    lab_out[:, :, 0] = L[0][:, :, 0]
    lab_out[:, :, 1:] = ab[0]
    rgb_out = lab2rgb(lab_out)

    # คืน PIL Image
    out_img = Image.fromarray((rgb_out * 255).astype(np.uint8))
    return out_img

@app.route('/colorize', methods=['POST'])
def colorize_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    try:
        pil_gray = Image.open(file.stream)
    except Exception as e:
        return jsonify({'error': f'Cannot open image: {e}'}), 400

    pil_color = colorize_image(pil_gray)

    buf = io.BytesIO()
    pil_color.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    # อย่าลืมติดตั้ง gdown ก่อนรัน: pip install gdown
    app.run(host='0.0.0.0', port=5000, debug=True)
