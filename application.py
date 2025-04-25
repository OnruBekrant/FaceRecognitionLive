"""
Face Recognition Web Application - Main Application
"""
import os
import logging
import cv2
import numpy as np
import time
import base64
import pickle
import io
from flask import Flask, render_template, Response, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Database setup
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

# Global variables
current_frame = None
recognition_threshold = 60  # Default similarity threshold

# Routes
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

def gen_frames():
    """
    Generator function to yield processed video frames for server-side simulation
    
    This is a fallback for when WebRTC camera access isn't available
    """
    global current_frame
    
    # For simplicity and to avoid worker timeouts, we'll use just a single static frame
    frame_width, frame_height = 640, 480
    
    try:
        # Create a basic static frame
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        frame[:] = (50, 70, 90)  # Simple background
        
        # Add face identification box and text
        x, y, w, h = 240, 160, 160, 160
        name = "John Doe"
        similarity = 85.5
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y+h-35), (x+w, y+h), (0, 255, 0), cv2.FILLED)
        label_text = f"{name} ({similarity:.1f}%)"
        cv2.putText(frame, label_text, (x+6, y+h-6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1)
        
        # Add some app text
        cv2.putText(frame, "Face Recognition System", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Simulated Camera", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
                   
        # Add the current time
        time_text = time.strftime('%H:%M:%S')
        cv2.putText(frame, f"Time: {time_text}", (frame_width - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Store the current frame for face capture
        current_frame = frame.copy()
        
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logger.error("Failed to encode frame")
            return
            
        # Convert to bytes and yield for streaming
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    except Exception as e:
        logger.error(f"Error in frame generation: {str(e)}")
        # Include the stack trace for better debugging
        import traceback
        logger.error(traceback.format_exc())

@app.route('/video_feed')
def video_feed():
    """Video streaming route for the webcam feed (server-side fallback)"""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process a frame captured from WebRTC camera"""
    global current_frame
    
    try:
        # Get frame from request
        if 'frame' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No frame found in request"
            }), 400
        
        # Read the image
        frame_file = request.files['frame']
        frame_bytes = frame_file.read()
        
        # Convert to numpy array
        frame_array = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({
                "status": "error",
                "message": "Failed to decode image"
            }), 400
        
        # Store the current frame for later use
        current_frame = frame.copy()
        
        # Process the frame to detect and recognize faces
        processed_frame, face_data = face_service.process_frame(frame)
        
        # Fix the JSON serialization by converting numpy types to Python native types
        detections = []
        for detection in face_data:
            detections.append({
                "name": detection["name"],
                "similarity": float(detection["similarity"]),
                "location": {
                    "top": int(detection["location"]["top"]),
                    "right": int(detection["location"]["right"]),
                    "bottom": int(detection["location"]["bottom"]),
                    "left": int(detection["location"]["left"])
                }
            })
        
        # Return the face detection data to the client
        return jsonify({
            "status": "success",
            "detections": detections
        })
        
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        }), 500

@app.route('/add_person', methods=['POST'])
def add_person():
    """Add a new person to the recognition database using server-side current frame"""
    global current_frame
    
    try:
        # Get name from request
        data = request.json
        person_name = data.get('name')
        
        if not person_name:
            return jsonify({"status": "error", "message": "Name is required"}), 400
        
        if current_frame is None:
            return jsonify({"status": "error", "message": "No camera frame available"}), 400
        
        # Add the face to the recognition service
        success = face_service.add_face(current_frame, person_name)
        
        if success:
            return jsonify({
                "status": "success", 
                "message": f"Person '{person_name}' added successfully"
            })
        else:
            return jsonify({
                "status": "error", 
                "message": "Failed to detect a face in the current frame"
            }), 400
            
    except Exception as e:
        logger.error(f"Error adding person: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": f"An error occurred: {str(e)}"
        }), 500

@app.route('/add_person_webcam', methods=['POST'])
def add_person_webcam():
    """Add a new person to the recognition database using client-side captured image"""
    try:
        # Get data from request
        data = request.json
        person_name = data.get('name')
        image_data = data.get('image_data')  # Base64 encoded image
        
        if not person_name:
            return jsonify({"status": "error", "message": "Name is required"}), 400
        
        if not image_data:
            return jsonify({"status": "error", "message": "Image data is required"}), 400
        
        # Decode the image
        try:
            img_bytes = base64.b64decode(image_data)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({"status": "error", "message": "Failed to decode image"}), 400
        except Exception as e:
            logger.error(f"Error decoding image: {str(e)}")
            return jsonify({"status": "error", "message": "Invalid image data"}), 400
        
        # Add the face to the recognition service
        success = face_service.add_face(frame, person_name)
        
        if success:
            return jsonify({
                "status": "success", 
                "message": f"Person '{person_name}' added successfully"
            })
        else:
            return jsonify({
                "status": "error", 
                "message": "Failed to detect a face in the image"
            }), 400
            
    except Exception as e:
        logger.error(f"Error adding person from webcam: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": f"An error occurred: {str(e)}"
        }), 500

@app.route('/get_faces', methods=['GET'])
def get_faces():
    """Get the list of known faces with thumbnails"""
    try:
        # Try to get faces from the database service first
        try:
            person_data = face_service.db_service.get_person_thumbnails()
            if person_data:
                return jsonify({
                    "status": "success",
                    "faces": person_data
                })
        except Exception as db_error:
            logger.error(f"Error getting faces from database: {str(db_error)}")
            # Fall back to in-memory data
        
        # If database retrieval failed or returned no data, use in-memory data
        faces = []
        for i, name in enumerate(face_service.known_face_names):
            face_data = {"id": i, "name": name}  # Use index as ID for in-memory data
            # Add thumbnail if available
            if (i < len(face_service.known_face_thumbnails) and 
                face_service.known_face_thumbnails[i] is not None):
                face_data["thumbnail"] = face_service.known_face_thumbnails[i]
            faces.append(face_data)
            
        return jsonify({
            "status": "success",
            "faces": faces
        })
    except Exception as e:
        logger.error(f"Error getting faces: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        }), 500
        
@app.route('/delete_person/<int:person_id>', methods=['DELETE'])
def delete_person(person_id):
    """Delete a person from the recognition database"""
    try:
        # Delete from database if it's available
        success = face_service.db_service.delete_person(person_id)
        
        if success:
            # Reload data to keep memory and database in sync
            face_service._load_face_data()
            
            return jsonify({
                "status": "success",
                "message": "Person deleted successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to delete person, person not found"
            }), 404
            
    except Exception as e:
        logger.error(f"Error deleting person: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        }), 500

@app.route('/update_threshold', methods=['POST'])
def update_threshold():
    """Update the recognition threshold"""
    global recognition_threshold
    
    try:
        data = request.json
        threshold = data.get('threshold')
        
        if threshold is None:
            return jsonify({"status": "error", "message": "Threshold is required"}), 400
        
        # Validate and update the threshold
        threshold = int(threshold)
        if threshold < 0 or threshold > 100:
            return jsonify({"status": "error", "message": "Threshold must be between 0 and 100"}), 400
        
        recognition_threshold = threshold
        logger.info(f"Recognition threshold updated to {threshold}%")
        
        return jsonify({
            "status": "success",
            "message": f"Threshold updated to {threshold}%"
        })
        
    except Exception as e:
        logger.error(f"Error updating threshold: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        }), 500

# For development use
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)