from application import db
import datetime

class Person(db.Model):
    """Person model to store face recognition data"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    thumbnail = db.Column(db.Text, nullable=True)  # Base64 encoded thumbnail image
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    # Faces associated with this person
    faces = db.relationship('Face', backref='person', lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<Person {self.name}>'


class Face(db.Model):
    """Face model to store face feature data for recognition"""
    id = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.Integer, db.ForeignKey('person.id'), nullable=False)
    face_data = db.Column(db.LargeBinary, nullable=False)  # Serialized face recognition features
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f'<Face for Person {self.person_id}>'