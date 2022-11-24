
from flask import Flask, jsonify, request
from flask_cors import CORS
import base64

app = Flask(__name__)
CORS(app)

from functions import df_top5_prediction_test_image

#define routes
@app.route("/upload",methods=["GET","POST"])
def upload():
    if request.method == 'POST':
        file = request.json
        base64_img_bytes = file.encode('utf-8')
        with open('decoded_image.jpeg', 'wb') as file_to_save:
            decoded_image_data = base64.decodebytes(base64_img_bytes)
            file_to_save.write(decoded_image_data)
        print("upload Completed")
        labels = list(df_top5_prediction_test_image('decoded_image.jpeg')['Label'])
        probs = list(df_top5_prediction_test_image('decoded_image.jpeg')['Probability'])
        jsonResp = []
        for i in range(len(labels)):
            pred = {
                "label":labels[i],
                "prob":probs[i]
            }
            jsonResp.append(pred)
        f = open('decoded_image.jpeg', "r+")
        f.truncate() 
        return jsonify(jsonResp)

if __name__=='__main__':
    app.run(debug=True, port=8002, host="0.0.0.0")