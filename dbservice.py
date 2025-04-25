import pickle
import logging
import base64
from models import Person, Face
from main import db

logger = logging.getLogger(__name__)

class DBFaceStorageService:
    """Database storage service for face recognition data"""
    
    def save_face(self, person_name, face_features, face_thumbnail=None):
        """
        Save a face to the database
        
        Args:
            person_name: Name of the person
            face_features: Numpy array of face features
            face_thumbnail: Base64 encoded thumbnail image
            
        Returns:
            Person: The created or updated person
        """
        try:
            # Check if person already exists
            person = Person.query.filter_by(name=person_name).first()
            
            if not person:
                # Create a new person
                person = Person(name=person_name)
                if face_thumbnail:
                    person.thumbnail = face_thumbnail
                db.session.add(person)
                db.session.commit()
                logger.info(f"Created new person: {person_name}")
            
            # Serialize face features
            serialized_features = pickle.dumps(face_features)
            
            # Create a new face
            face = Face(person_id=person.id, face_data=serialized_features)
            db.session.add(face)
            db.session.commit()
            
            logger.info(f"Added face for person: {person_name}")
            return person
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error saving face: {str(e)}")
            raise
    
    def get_all_faces(self):
        """
        Get all faces from the database
        
        Returns:
            List of tuples containing (person_name, face_features_array, thumbnail)
        """
        try:
            faces = []
            
            # Query all faces with their related person data
            face_records = db.session.query(Face, Person).join(Person).all()
            
            for face_record, person in face_records:
                # Deserialize the face features
                face_features = pickle.loads(face_record.face_data)
                # Add to the list
                faces.append((person.name, face_features, person.thumbnail))
            
            logger.info(f"Retrieved {len(faces)} faces from database")
            return faces
            
        except Exception as e:
            logger.error(f"Error getting faces: {str(e)}")
            return []
    
    def get_person_names(self):
        """
        Get list of all person names
        
        Returns:
            List of person names
        """
        try:
            persons = Person.query.all()
            return [person.name for person in persons]
        except Exception as e:
            logger.error(f"Error getting person names: {str(e)}")
            return []
    
    def get_person_thumbnails(self):
        """
        Get list of person names with their thumbnails
        
        Returns:
            List of dictionaries with name and thumbnail
        """
        try:
            persons = Person.query.all()
            return [{"name": person.name, "thumbnail": person.thumbnail} for person in persons]
        except Exception as e:
            logger.error(f"Error getting person thumbnails: {str(e)}")
            return []