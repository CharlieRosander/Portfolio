from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session
from datetime import datetime
import requests
import base64
from io import BytesIO
from PIL import Image
import dotenv
import os
import openai

dotenv.load_dotenv()

app = Flask(__name__)
app.secret_key = "secret_key"  # Replace with a secure key in production.
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///grocery_list.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Flask-Session configuration
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

openai_api_key = os.getenv("OPENAI_API_KEY")

db = SQLAlchemy(app)

# Database Models
class Category(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    order = db.Column(db.Integer, nullable=False)

class GroceryItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    category_id = db.Column(db.Integer, db.ForeignKey("category.id"), nullable=False)

@app.route("/")
def index():
    categories = Category.query.order_by(Category.order).all()
    return render_template("index.html", categories=categories)

@app.route("/add_category", methods=["GET", "POST"])
def add_category():
    if request.method == "POST":
        name = request.form["name"]
        order = request.form["order"]
        category = Category(name=name, order=order)
        db.session.add(category)
        db.session.commit()
        flash("Category added successfully!")
        return redirect(url_for("index"))
    return render_template("add_category.html")

@app.route("/upload_list", methods=["GET", "POST"])
def upload_list():
    if request.method == "POST":
        if 'preview' in request.form:
            file = request.files["grocery_list"]
            if file:
                image = Image.open(file)
                if image.mode == "RGBA":
                    image = image.convert("RGB")
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                session['img_str'] = img_str
                return render_template("upload_list.html", img_str=img_str)
        elif 'submit' in request.form:
            img_str = session.get('img_str')
            if img_str:
                # Get categories and their order
                categories = Category.query.order_by(Category.order).all()
                category_order = [category.name for category in categories]

                # Call OpenAI API
                organized_list = organize_grocery_list(img_str, category_order)
                if organized_list:
                    flash("List processed successfully!")
                    return render_template(
                        "organized_list.html", organized_list=organized_list
                    )
                else:
                    flash("An error occurred while processing the list.")
                    return redirect(url_for("upload_list"))

    return render_template("upload_list.html")

def organize_grocery_list(image_data, category_order):
    response = openai.chat.completions.create(
        model="gpt-4o",  # Update the model based on your subscription
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that organizes grocery lists.",
            },
            {
                "role": "user",
                "content": f"Here is the grocery list image (base64 encoded): {image_data}.\n\n"
                f"The categories in order of the store layout are: {', '.join(category_order)}.\n\n"
                "Please extract the items from the image and organize them according to the given category order.",
            },
        ],
        temperature=0.5,
    )

    if response:
        organized_list = response["choices"][0]["message"]["content"]
        return organized_list
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

@app.route("/edit_category/<int:id>", methods=["GET", "POST"])
def edit_category(id):
    category = Category.query.get_or_404(id)
    if request.method == "POST":
        category.name = request.form["name"]
        category.order = request.form["order"]
        db.session.commit()
        flash("Category updated successfully!")
        return redirect(url_for("index"))
    return render_template("edit_category.html", category=category)

@app.route("/delete_category/<int:id>", methods=["POST"])
def delete_category(id):
    category = Category.query.get_or_404(id)
    db.session.delete(category)
    db.session.commit()
    flash("Category deleted successfully!")
    return redirect(url_for("index"))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
