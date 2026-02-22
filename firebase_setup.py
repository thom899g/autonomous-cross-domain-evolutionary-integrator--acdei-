"""
Firebase Admin SDK setup and management for ACDEI
Handles initialization, authentication, and connection management
"""
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import json

import firebase_admin
from firebase_admin import credentials, firestore, initialize_app
from firebase_admin.exceptions import FirebaseError
from google.cloud.firestore_v1.client import Client as FirestoreClient

from config import config

logger = logging.getLogger(__name__)

class FirebaseManager:
    """Manages Firebase connection and provides Firestore client"""
    
    _instance = None
    _app = None
    _db = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FirebaseManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = False
            self.logger = logging.getLogger(f"{__name__