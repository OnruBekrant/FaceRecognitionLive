"""
Main module for the Flask application
This is a simple entrypoint for gunicorn
"""
from application import app

# Make app available to gunicorn
# This is required for the workflow to work properly
# as it uses main:app as the WSGI entry point