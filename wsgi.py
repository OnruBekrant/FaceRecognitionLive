"""
WSGI entry point for the face recognition application
"""
import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

# Create database instance
db = SQLAlchemy(model_class=Base)

# Create the Flask app
app = Flask(__name__)

# Setup a secret key, required by sessions
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or "a secret key"

# Get database URL from environment
database_url = os.environ.get("DATABASE_URL")
if not database_url:
    logger.error("DATABASE_URL environment variable not set!")
else:
    logger.info(f"Using database URL: {database_url.split('@')[0]}@...")

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize the app with the extension, flask-sqlalchemy >= 3.0.x
db.init_app(app)

# Initialize models and services
with app.app_context():
    # Import models and create tables
    import models
    db.create_all()
    
    # Initialize services
    from dbservice import DBFaceStorageService
    from face_recognition_service import FaceRecognitionService
    
    # Initialize face recognition service
    face_service = FaceRecognitionService()
    
    # Initialize database service and attach to face service
    db_service = DBFaceStorageService()
    face_service.db_service = db_service

# Import routes and register them
from routes import register_routes
register_routes(app, face_service)

# For development use
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)