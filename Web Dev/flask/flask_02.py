# Tutorial 2 : Rendering Html templates and using python functions inside Html
from flask import Flask, render_template

app = Flask(__name__)

# 1. render html
@app.route("/")
def home():
    return render_template("index.html")

# 2. passing variables
@app.route("/<name>/")
def user_name(name):
    return render_template("user.html", user_name=name, age=30)

# 3. for loop : odd numbers
@app.route("/odd_numbers_<int:number>/")
def odd_numbers(number):
    return render_template("odd_numbers.html", limit=number)
    
# 4. list
@app.route("/name_list/")
def name_list():
    return render_template("name_list.html", names=["jim", "tim", "carry"])
    
if __name__ == "__main__":
    app.run()