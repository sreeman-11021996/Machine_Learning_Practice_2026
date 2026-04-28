# Tutorial 3: Template inheritence & Bootstrap

# Bootstrap : 
"""
    Webpage = https://getbootstrap.com/docs/4.3/getting-started/introduction/
        -> css link in getbootstrap website to be copied before the title tag inside the <head> 
        -> js links in getbootstrap website to be copied at the end inside the <body>
        -> NavBar code from the website 
"""

from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index_inherit_css.html", content = "Testing css bootstrap")

@app.route("/<name>")
def user(name):
    return render_template("user_inherit_css.html", user = name)

@app.route("/normal_home")
def normal_home():
    return render_template("index_inherit.html", content = "Testing Inheritence")

if __name__ == "__main__":
    app.run(debug=True)