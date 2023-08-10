import subprocess
from flask import Flask, request, jsonify, render_template, url_for
import os
import shutil

app = Flask(__name__, static_url_path='/static')

def run_yolo_prediction(image_path):
    yolo_model_path = "yolo_model/best.pt"
    temp_result_dir = os.path.join('static', 'results')
    yolo_command = f'yolo task=detect mode=predict model={yolo_model_path} project={temp_result_dir} conf=0.25 source={image_path} save=True'

    os.makedirs(temp_result_dir, exist_ok=True)

    try:
        subprocess.run(yolo_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        return "Error running YOLO prediction.", None

    result_image_path = os.path.join(temp_result_dir, 'uploaded_image.jpg')
    predict_dirs = [f for f in os.listdir(temp_result_dir) if f.startswith('predict')]
    latest_predict_dir = max(predict_dirs, key=lambda x: int(x[len('predict'):]) if x[len('predict'):] else -1, default=None)

    if latest_predict_dir:
        result_image_path = os.path.join(temp_result_dir, latest_predict_dir, 'uploaded_image.jpg')

        predicted_images_dir = os.path.join('results', latest_predict_dir)
        os.makedirs(predicted_images_dir, exist_ok=True)

        predicted_image_dest = os.path.join(predicted_images_dir, 'uploaded_image.jpg')
        shutil.copy(result_image_path, predicted_image_dest)

        result_url = url_for('static', filename=predicted_image_dest.replace("\\", "/"))

        return result_url
    else:
        return None
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    image_path = "temp_images/uploaded_image.jpg"

    image.save(image_path)

    result_url = run_yolo_prediction(image_path)

    return render_template('predict.html', result_url=result_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
