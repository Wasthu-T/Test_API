from flask import Flask

# Inisiasi Flask
app = Flask(__name__)

from app import routes
