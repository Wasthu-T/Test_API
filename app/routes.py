from app import app, response, limiter
from app.controller import aicontroller
from flask import request

@app.route('/api/predict', methods=["POST"])
@limiter.limit("10 per minute")
def predict() : 
    if request.method == 'POST':
        return aicontroller.predict()
    else:
        return response.badRequest(None, 'Terjadi kesalahan')

@app.route('/')
def home():
    return "welcome"
