from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    send_file,
)
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
from db_schema import ImageModel, db
import io
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.secret_key = "secret"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = "uploads"
db.init_app(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/create_category")
def create_category():
    return render_template("create_category.html")


@app.route("/upload_image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        image_file = request.files["image_file"]
        clothing_name = request.form.get("clothing_name")
        clothing_size = request.form.get("clothing_size")
        clothing_owner = request.form.get("clothing_owner")

        if image_file:
            filename = secure_filename(image_file.filename)

            # Save the image to a temporary file
            temp_file = io.BytesIO()
            image_file.save(temp_file)
            temp_file.seek(0)

            # Read the image data from the temporary file
            blob_data = temp_file.read()

            # Create a new image entry
            new_image = ImageModel(
                image_name=filename,
                image_blob=blob_data,
                clothing_name=clothing_name,
                clothing_size=clothing_size,
                clothing_owner=clothing_owner,
            )
            db.session.add(new_image)
            db.session.commit()

            flash("Image uploaded successfully", "success")
            return redirect(url_for("index"))

        flash("Failed to upload image", "danger")
    return render_template("upload_image.html")


@app.route("/search_images", methods=["GET", "POST"])
def search_images():
    images = None
    if request.method == "POST":
        search_query = request.form.get("search_query")
        if search_query:
            # Perform the search query on the database
            images = ImageModel.query.filter(
                (ImageModel.image_name.contains(search_query))
                | (ImageModel.clothing_name.contains(search_query))
                | (ImageModel.clothing_owner.contains(search_query))
            ).all()

            if not images:
                flash("No images found for your search query.", "warning")
    return render_template("search_images.html", images=images)


@app.route("/display_image/<int:image_id>")
def display_image(image_id):
    image = ImageModel.query.get(image_id)
    if image:
        return send_file(
            io.BytesIO(image.image_blob),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name=image.image_name,
        )
    else:
        flash("Image not found", "danger")
        return redirect(url_for("index"))


@app.route("/load_image")
def load_image(image_id, output_path):
    # Hämta bilden från databasen
    image = db.query(ImageModel).get(image_id)
    if image:
        # Skriv binärdata till en fil
        with open(output_path, "wb") as file:
            file.write(image.data)
    return redirect(url_for("index"))


@app.route("/delete_image")
def delete_image():
    return "Delete Image"


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
