import requests
import base64
url = "http://0.0.0.0:31337"  # TODO server 1
print(url)
image_file = 'stills/speaker01.png'

with open(image_file, "rb") as f:
    im_bytes = f.read()
im_b64 = base64.b64encode(im_bytes).decode("utf8")

try:
    print(url + "/generate")
    response = requests.post(url + "/generate", json={
        "text": "Here you go, a link for Biondo Racing Products and other related pages.",
        "pca_weights": [
            [0] * 10
        ],
        "model_name": 'vctk-vits',
        "key": "test.wav",
        "reference_image": im_b64,
        "normalize_embedding": True
    })
    output_path = 'test.mp4'
    with open(output_path, 'wb') as f:
        f.write(response.content)
except Exception as e:
    print(e)
