import base64
from flask import Flask, render_template, jsonify, request
from camera import process_frame, music_rec

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    data = request.get_json()
    # Strip the data-URL prefix (e.g. "data:image/jpeg;base64,...")
    _, encoded = data['frame'].split(',', 1)
    image_bytes = base64.b64decode(encoded)

    frame_bytes, emotion, songs_df = process_frame(image_bytes)

    result = {
        'emotion': emotion,
        'songs': songs_df.to_dict(orient='records'),
    }
    if frame_bytes:
        result['frame'] = 'data:image/jpeg;base64,' + base64.b64encode(frame_bytes).decode()

    return jsonify(result)


@app.route('/t')
def gen_table():
    return music_rec().to_json(orient='records')


if __name__ == '__main__':
    app.debug = True
    app.run()
