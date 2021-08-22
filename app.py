from tensorflow.keras.models import load_model
from typing import *
import os
from flask import Flask, request
from selenium.webdriver.chrome.options import Options 
import chromedriver_autoinstaller
from selenium import webdriver
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model("/app/model/test_640_360_3-2.h5")

def get_config() -> Options:
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=1280,720")
    return chrome_options

chromedriver_autoinstaller.install()
driver = webdriver.Chrome(chrome_options=get_config())

def dump_printscreen(driver: webdriver.Chrome, url: str) -> str:  
    driver.get(url)
    file_path = f"web_page.png"
    driver.get_screenshot_as_file(file_path)
    return str(file_path)

def image_file_to_numpy(file_path: str):
    img = image.load_img(file_path, target_size=(640,360))
    img = image.img_to_array(img)
    img /= 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(img_numpy):
    result = model.predict(img_numpy)[0]
    d = {
        "200": f"{result[0]:3f}",
        "!200": f"{result[1]:3f}",
    }
    return d

@app.route("/", methods=["POST"])
def endpoint():
    url = request.get_data(as_text=True)
    print(url)
    image_path = dump_printscreen(driver, url)
    image_array = image_file_to_numpy(image_path)
    d = predict(image_array)

    return d


if __name__ == "__main__":
    
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 30001)))