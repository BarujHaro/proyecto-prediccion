from flask import Flask
from views import views
from waitress import serve

app = Flask(__name__)
app.register_blueprint(views, url_prefix="/")


if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=5000)
