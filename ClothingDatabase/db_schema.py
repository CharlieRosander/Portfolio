# from sqlalchemy import Column, Integer, String, LargeBinary
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


# Definiera en ORM-modell för bilder
class ImageModel(db.Model):
    __tablename__ = "images"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    image_name = db.Column(db.String, nullable=False)
    image_blob = db.Column(db.LargeBinary, nullable=False)
    clothing_name = db.Column(db.String, nullable=True)
    clothing_size = db.Column(db.Integer, nullable=True)
    clothing_owner = db.Column(db.String, nullable=True)
