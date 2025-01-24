from app import app, response
from app.controller import aicontroller
from flask import request

@app.route('/api/predict', methods=["POST"])
def predict() : 
    if request.method == 'POST':
        return aicontroller.predict()
    else:
        return response.badRequest(None, 'Terjadi kesalahan')

@app.route('/')
def home():
    return "welcome"
