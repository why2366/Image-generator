from flask import Flask, request, jsonify, send_file
import base64
import io
from PIL import Image
import requests
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)
# Stable Diffusion 本地 API 地址
SD_API_URL = "http://121.4.44.52:90"

def encode_file_to_base64(file_path):
    """将文件编码为 Base64"""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

@app.route('/upload', methods=['POST'])
def upload_image():
    """接收用户上传的图片，反推提示词并生成图片"""
    
    if 'image' not in request.files:
        return jsonify({"error": "未上传图片"}), 400

    # 读取上传的图片并编码为 Base64
    image_file = request.files['image']
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    # 调用提示词反推接口
    payload_for_clip = {
        "image": encoded_image,
        "model": "deepdanbooru"
    }
    response = requests.post(f'{SD_API_URL}/sdapi/v1/interrogate', json=payload_for_clip)
    if response.status_code != 200:
        return jsonify({"error": "提示词反推失败"}), 500
    caption = response.json().get('caption', '')

    # 调用图生图接口
    negative_prompt = " lowres, text, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, text, signature, watermark, simple background, toony, dated, low res, line art, flat colors, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    payload_for_generation = {
        "prompt": caption,
        "negative_prompt": negative_prompt,
        "steps": 20,
        "sampler_name": "DPM++ 2M",
        "width": 800,
        "height": 800,
        "batch_size": 1,
        "n_iter": 1,
        "seed": -1,
        "cfg_scale": 7,
        "restore_faces": True,
        "init_images": [encoded_image],
        "denoising_strength": 0.7
    }
    gen_response = requests.post(f'{SD_API_URL}/sdapi/v1/img2img', json=payload_for_generation)
    if gen_response.status_code != 200:
        return jsonify({"error": "图像生成失败"}), 500

    # 解码生成的图像
    gen_image_data = base64.b64decode(gen_response.json()['images'][0])
    image = Image.open(io.BytesIO(gen_image_data))

    # 保存生成的图像
    output_path = "C:/Users/Administrator/Desktop/image-saved"
    image.save(output_path)
    return send_file(output_path, mimetype='image/png')
    pass

if __name__ == '__main__':
    # 允许局域网访问
    app.run(host='0.0.0.0', port=5000)
