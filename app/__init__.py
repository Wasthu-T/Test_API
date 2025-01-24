from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Inisiasi Flask
app = Flask(__name__)
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

from app import routes