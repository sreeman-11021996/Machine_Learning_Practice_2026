# Tutorial 1 : Basic web page set up, passing variables into webpage and redirecting for 
# say admin acess

from flask import Flask, redirect, url_for

application = Flask(__name__)

# 1. home page
@application.route("/")
def home():
    return "Hello! This is the main page. <h1>Hello</h1>"

# 2. user page
@application.route("/<name>")
def user(name):
    return f"Hello {name}"
    
# 3. admin page - restrict access
# /admin/ -> "localhost/admin" and "localhost/admin/" both work now
@application.route("/admin/")
def admin():
    return (redirect(url_for("user", name="Admin!")))


if __name__ == "__main__":
    application.run()