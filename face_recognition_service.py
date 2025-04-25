import cv2
import numpy as np
import os
import logging
import pickle
import base64
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class FaceRecognitionService:
    """Service for detecting and recognizing faces in video frames using OpenCV"""
    
    def __init__(self):
        # Directory for storing face data
        self.face_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_data')
        os.makedirs(self.face_data_dir, exist_ok=True)
        
        # File to store face data
        self.face_data_file = os.path.join(self.face_data_dir, 'face_data.pkl')
        
        # We'll initialize the database service later to avoid circular imports
        self.db_service = None
        
        # Initialize known faces database
        self.known_face_features = []
        self.known_face_names = []
        self.known_face_thumbnails = []  # Store small thumbnails of faces
        
        # Load face recognizer
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Try to load existing model
        self._load_face_data()
        
        # Frame processing parameters
        self.process_every_n_frames = 1  # Process every frame for smoother detection
        self.frame_count = 0
        self.min_face_size = (60, 60)  # Smaller minimum face size to detect faces further from camera
        self.scale_factor = 1.05  # Smaller scale factor for more accurate but slower detection
        self.min_neighbors = 4  # Lower minimum neighbors for more sensitive detection
        
        logger.info("Face recognition service initialized")
        
    def _load_face_data(self):
        """Load existing face data from the database"""
        try:
            # Load from database if it's initialized
            if self.db_service is not None:
                try:
                    # Get all faces from database
                    db_faces = self.db_service.get_all_faces()
                    
                    # Reset current data
                    self.known_face_features = []
                    self.known_face_names = []
                    self.known_face_thumbnails = []
                    
                    # Populate from database
                    for name, features, thumbnail in db_faces:
                        self.known_face_features.append(features)
                        self.known_face_names.append(name)
                        self.known_face_thumbnails.append(thumbnail)
                    
                    # If we have face data from the database, train the recognizer
                    if self.known_face_features and len(self.known_face_features) > 0:
                        labels = np.array([i for i in range(len(self.known_face_names))])
                        self.face_recognizer.train(self.known_face_features, labels)
                        logger.info(f"Loaded {len(self.known_face_names)} faces from database")
                        return
                    
                except Exception as db_error:
                    logger.error(f"Error loading from database, falling back to file: {str(db_error)}")
            else:
                logger.info("Database service not initialized, using file storage")
            
            # Fall back to file-based storage if database failed or isn't initialized
            if os.path.exists(self.face_data_file):
                with open(self.face_data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_features = data.get('features', [])
                    self.known_face_names = data.get('names', [])
                    self.known_face_thumbnails = data.get('thumbnails', [])
                    
                    # If no thumbnails but we have features, generate empty thumbnails
                    if len(self.known_face_thumbnails) == 0 and len(self.known_face_features) > 0:
                        self.known_face_thumbnails = [None] * len(self.known_face_features)
                    
                # If we have face data, train the recognizer
                if self.known_face_features and len(self.known_face_features) > 0:
                    labels = np.array([i for i in range(len(self.known_face_names))])
                    self.face_recognizer.train(self.known_face_features, labels)
                    logger.info(f"Loaded {len(self.known_face_names)} faces from file")
                else:
                    logger.info("No existing face data found in file")
            else:
                logger.info("No existing face data file found")
        except Exception as e:
            logger.error(f"Error loading face data: {str(e)}")
    
    def _save_face_data(self):
        """Save face data to file"""
        try:
            data = {
                'features': self.known_face_features,
                'names': self.known_face_names,
                'thumbnails': self.known_face_thumbnails
            }
            with open(self.face_data_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved {len(self.known_face_names)} faces to database")
        except Exception as e:
            logger.error(f"Error saving face data: {str(e)}")
    
    def add_face(self, face_image: np.ndarray, person_name: str) -> bool:
        """
        Add a new face to the known faces database
        
        Args:
            face_image: Image containing a face
            person_name: Name of the person
            
        Returns:
            bool: True if face was added successfully, False otherwise
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_face_size
            )
            
            if len(faces) == 0:
                logger.warning("No face detected in the provided image")
                return False
                
            # Use the first face found
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize for consistency
            face_roi = cv2.resize(face_roi, (100, 100))
            
            # Store a thumbnail of the face
            face_color = face_image[y:y+h, x:x+w]
            thumbnail = cv2.resize(face_color, (64, 64))
            # Convert to base64 for storage and transmission
            _, buffer = cv2.imencode('.jpg', thumbnail)
            thumbnail_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Save to database first if it's initialized
            if self.db_service is not None:
                try:
                    self.db_service.save_face(person_name, face_roi, thumbnail_b64)
                    logger.info(f"Saved face for {person_name} to database")
                except Exception as db_error:
                    logger.error(f"Error saving to database: {str(db_error)}")
                    # Continue even if database save fails
            
            # Add to the in-memory database
            self.known_face_features.append(face_roi)
            self.known_face_names.append(person_name)
            self.known_face_thumbnails.append(thumbnail_b64)
            
            # Train the recognizer with all faces
            if len(self.known_face_features) > 0:
                labels = np.array([i for i in range(len(self.known_face_names))])
                self.face_recognizer.train(self.known_face_features, labels)
            
            # Save to the file as backup
            self._save_face_data()
            
            logger.info(f"Added face for {person_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding face: {str(e)}")
            return False
    
    def compare_faces(self, face_id1, face_id2):
        """
        Compare two faces and calculate their similarity
        
        Args:
            face_id1: ID of the first face
            face_id2: ID of the second face
            
        Returns:
            dict: A dictionary with comparison data including similarity percentage
        """
        try:
            # Get face encodings for both faces
            # If using DB service, retrieve from database
            if len(self.known_face_features) <= max(face_id1, face_id2):
                raise ValueError(f"Invalid face IDs: {face_id1}, {face_id2}")
                
            face1_encoding = self.known_face_features[face_id1]
            face2_encoding = self.known_face_features[face_id2]
            
            # Get names
            face1_name = self.known_face_names[face_id1]
            face2_name = self.known_face_names[face_id2]
            
            # Get thumbnails if available
            face1_thumbnail = None
            if face_id1 < len(self.known_face_thumbnails):
                face1_thumbnail = self.known_face_thumbnails[face_id1]
                
            face2_thumbnail = None
            if face_id2 < len(self.known_face_thumbnails):
                face2_thumbnail = self.known_face_thumbnails[face_id2]
            
            # Calculate similarity (1 - distance)
            # Convert from OpenCV feature distance to percentage similarity
            distance = np.linalg.norm(face1_encoding - face2_encoding)
            # Convert distance to similarity percentage (closer to 0 = more similar)
            # Use a formula that maps typical distances to reasonable percentages
            similarity = max(0, min(100, 100 * (1 - distance / 1.0)))
            
            # Return comparison data
            return {
                "face1": {
                    "id": face_id1,
                    "name": face1_name,
                    "thumbnail": face1_thumbnail
                },
                "face2": {
                    "id": face_id2,
                    "name": face2_name,
                    "thumbnail": face2_thumbnail
                },
                "similarity": similarity,
                "distance": distance
            }
            
        except Exception as e:
            logger.error(f"Error comparing faces: {str(e)}")
            raise
            
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process a video frame to detect and recognize faces
        
        Args:
            frame: The video frame to process
            
        Returns:
            Tuple containing:
                - Processed frame with face annotations
                - List of dictionaries with face data
        """
        # Initialize face data list
        face_data = []
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Only process every nth frame to improve performance
        if self.frame_count % self.process_every_n_frames == 0:
            # Detect faces in the current frame
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_face_size
            )
            
            # Process each face
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                
                # Resize for recognition
                face_roi_resized = cv2.resize(face_roi, (100, 100))
                
                # Default values
                name = "Unknown"
                similarity = 0.0
                
                # Only try to recognize if we have trained faces
                if len(self.known_face_names) > 0:
                    try:
                        # Recognize the face
                        label, confidence = self.face_recognizer.predict(face_roi_resized)
                        
                        # Lower confidence means better match in OpenCV
                        # Convert to similarity score (0-100%)
                        similarity = max(0, min(100, 100 - confidence / 2))
                        
                        # Use a threshold to determine if it's a match
                        if similarity >= 50:  # Düşürülmüş 50% similarity threshold
                            name = self.known_face_names[label]
                    except Exception as e:
                        logger.error(f"Error in face recognition: {str(e)}")
                
                # Draw a box around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Create label with name and similarity
                label_text = f"{name} ({similarity:.1f}%)" if name != "Unknown" else "Unknown"
                
                # Draw a filled rectangle for the label background
                cv2.rectangle(frame, (x, y+h-35), (x+w, y+h), (0, 255, 0), cv2.FILLED)
                
                # Add the label
                cv2.putText(frame, label_text, (x+6, y+h-6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1)
                
                # Add face data to the list
                # Convert numpy int32/float types to Python native types for JSON serialization
                face_data.append({
                    "name": name,
                    "similarity": float(similarity),
                    "location": {
                        "top": int(y),
                        "right": int(x+w),
                        "bottom": int(y+h),
                        "left": int(x)
                    }
                })
        
        self.frame_count += 1
        
        return frame, face_data
