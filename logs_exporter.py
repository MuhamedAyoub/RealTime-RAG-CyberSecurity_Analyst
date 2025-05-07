from flask import Flask, request
import logging

app = Flask(__name__)

# Configure custom access logger
access_log = logging.getLogger('access')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(remote_addr)s - %(user)s [%(method)s %(path)s] "%(status)s" "%(user_agent)s"')
handler.setFormatter(formatter)
access_log.addHandler(handler)
access_log.setLevel(logging.INFO)

@app.after_request
def log_request(response):
    access_log.info('', extra={
        'remote_addr': request.remote_addr,
        'user': request.authorization.username if request.authorization else 'guest',
        'method': request.method,
        'path': request.path,
        'status': f"{response.status_code} {response.status}",
        'user_agent': request.headers.get('User-Agent')
    })
    return response

@app.route('/')
def home():
    return "Welcome to Flask App"

@app.route('/login', methods=["PUT"])
def login():
    return "Login failed", 404

@app.route('/api/data', methods=["PUT"])
def data():
    return "Error", 500

@app.route('/index.html', methods=["GET", "DELETE"])
def index():
    if request.method == "DELETE":
        return "Unauthorized", 401
    return "OK", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
