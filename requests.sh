curl -X PUT http://localhost:5000/login \
  -H "Authorization: Basic Ym9iOnBhc3N3b3Jk" \
  -A "curl/7.68.0"

curl -X PUT http://localhost:5000/api/data \
  -H "Authorization: Basic Z3Vlc3Q6cGFzc3dvcmQ=" \
  -A "Python-urllib/3.8"

curl -X PUT http://localhost:5000/api/data \
  -H "Authorization: Basic am9objpwYXNzd29yZA==" \
  -A "PostmanRuntime/7.26.8"

curl -X DELETE http://localhost:5000/index.html \
  -H "Authorization: Basic Z3Vlc3Q6cGFzc3dvcmQ=" \
  -A "curl/7.68.0"

curl http://localhost:5000/index.html \
  -H "Authorization: Basic Ym9iOnBhc3N3b3Jk" \
  -A "curl/7.68.0"
