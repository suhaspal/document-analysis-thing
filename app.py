import os
from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from pymongo import MongoClient
from bson.objectid import ObjectId
import base64
from pdf2image import convert_from_path
import fitz
import docx2txt
from io import BytesIO
import logging
from datetime import timedelta

from document_pipeline import document_pipeline, upload_and_parse_document

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'your-secret-key'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
jwt = JWTManager(app)
bcrypt = Bcrypt(app)
socketio = SocketIO(app, cors_allowed_origins="*")
client = MongoClient('mongodb://localhost:27017/')
db = client['annotation_db']

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def render_document(filename):
    file_extension = os.path.splitext(filename)[1].lower()
    
    if file_extension == '.pdf':
        images = convert_from_path(filename)
        rendered_pages = []
        for i, image in enumerate(images):
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            rendered_pages.append(f"data:image/png;base64,{img_str}")
        return rendered_pages
    
    elif file_extension in ['.docx', '.doc']:
        text = docx2txt.process(filename)
        return [text]
    
    elif file_extension == '.txt':
        with open(filename, 'r') as file:
            text = file.read()
        return [text]
    
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400
    if db.users.find_one({'username': username}):
        return jsonify({'error': 'Username already exists'}), 400
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    user_id = db.users.insert_one({'username': username, 'password': hashed_password}).inserted_id
    return jsonify({'message': 'User registered successfully', 'userId': str(user_id)}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    user = db.users.find_one({'username': username})
    if user and bcrypt.check_password_hash(user['password'], password):
        access_token = create_access_token(identity=str(user['_id']))
        return jsonify({'access_token': access_token}), 200
    return jsonify({'error': 'Invalid username or password'}), 401

@app.route('/upload', methods=['POST'])
@jwt_required()
def upload_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        try:
            text, final_vec, vector_db = document_pipeline(filename)
            rendered_pages = render_document(filename)
            doc_id = db.documents.insert_one({
                'filename': file.filename,
                'text': text,
                'vector': final_vec.tolist(),
                'rendered_pages': rendered_pages,
                'user_id': get_jwt_identity()
            }).inserted_id
            return jsonify({'documentId': str(doc_id), 'text': text, 'renderedPages': rendered_pages}), 200
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return jsonify({'error': 'Error processing document'}), 500

@app.route('/document/<document_id>', methods=['GET'])
@jwt_required()
def get_document(document_id):
    try:
        document = db.documents.find_one({'_id': ObjectId(document_id), 'user_id': get_jwt_identity()})
        if document:
            return jsonify({
                'id': str(document['_id']),
                'filename': document['filename'],
                'text': document['text'],
                'renderedPages': document['rendered_pages']
            })
        return jsonify({'error': 'Document not found'}), 404
    except Exception as e:
        logger.error(f"Error retrieving document: {str(e)}")
        return jsonify({'error': 'Error retrieving document'}), 500

@app.route('/annotations/<document_id>', methods=['GET'])
@jwt_required()
def get_annotations(document_id):
    try:
        annotations = list(db.annotations.find({'document_id': document_id}))
        return jsonify([{**ann, '_id': str(ann['_id'])} for ann in annotations])
    except Exception as e:
        logger.error(f"Error retrieving annotations: {str(e)}")
        return jsonify({'error': 'Error retrieving annotations'}), 500

@app.route('/annotations', methods=['POST'])
@jwt_required()
def create_annotation():
    data = request.get_json()
    try:
        annotation = {
            'document_id': data['document_id'],
            'user_id': get_jwt_identity(),
            'type': data['type'],
            'content': data['content'],
            'position': data['position']
        }
        result = db.annotations.insert_one(annotation)
        return jsonify({'id': str(result.inserted_id)}), 201
    except Exception as e:
        logger.error(f"Error creating annotation: {str(e)}")
        return jsonify({'error': 'Error creating annotation'}), 500

@app.route('/annotations/<annotation_id>', methods=['PUT'])
@jwt_required()
def update_annotation(annotation_id):
    data = request.get_json()
    user_id = get_jwt_identity()
    try:
        result = db.annotations.update_one(
            {'_id': ObjectId(annotation_id), 'user_id': user_id},
            {'$set': {
                'type': data.get('type'),
                'content': data.get('content'),
                'position': data.get('position')
            }}
        )
        if result.modified_count:
            updated_annotation = db.annotations.find_one({'_id': ObjectId(annotation_id)})
            return jsonify({**updated_annotation, '_id': str(updated_annotation['_id'])}), 200
        else:
            return jsonify({'error': 'Annotation not found or you do not have permission to update it'}), 404
    except Exception as e:
        logger.error(f"Error updating annotation: {str(e)}")
        return jsonify({'error': 'Error updating annotation'}), 500

@app.route('/annotations/<annotation_id>', methods=['DELETE'])
@jwt_required()
def delete_annotation(annotation_id):
    user_id = get_jwt_identity()
    try:
        result = db.annotations.delete_one({'_id': ObjectId(annotation_id), 'user_id': user_id})
        if result.deleted_count:
            return jsonify({'message': 'Annotation deleted successfully'}), 200
        else:
            return jsonify({'error': 'Annotation not found or you do not have permission to delete it'}), 404
    except Exception as e:
        logger.error(f"Error deleting annotation: {str(e)}")
        return jsonify({'error': 'Error deleting annotation'}), 500

@socketio.on('join')
def on_join(data):
    room = data['document_id']
    join_room(room)
    emit('status', {'msg': f"{get_jwt_identity()} has joined the document."}, room=room)

@socketio.on('leave')
def on_leave(data):
    room = data['document_id']
    leave_room(room)
    emit('status', {'msg': f"{get_jwt_identity()} has left the document."}, room=room)

@socketio.on('new_annotation')
def handle_new_annotation(data):
    try:
        annotation = {
            'document_id': data['document_id'],
            'user_id': get_jwt_identity(),
            'type': data['type'],
            'content': data['content'],
            'position': data['position']
        }
        result = db.annotations.insert_one(annotation)
        data['id'] = str(result.inserted_id)
        emit('annotation_update', data, room=data['document_id'])
    except Exception as e:
        logger.error(f"Error handling new annotation: {str(e)}")
        emit('error', {'message': 'Error creating annotation'})

@socketio.on('update_annotation')
def handle_update_annotation(data):
    try:
        user_id = get_jwt_identity()
        annotation_id = data['id']
        result = db.annotations.update_one(
            {'_id': ObjectId(annotation_id), 'user_id': user_id},
            {'$set': {
                'type': data.get('type'),
                'content': data.get('content'),
                'position': data.get('position')
            }}
        )
        if result.modified_count:
            updated_annotation = db.annotations.find_one({'_id': ObjectId(annotation_id)})
            emit('annotation_updated', {**updated_annotation, '_id': str(updated_annotation['_id'])}, room=data['document_id'])
        else:
            emit('error', {'message': 'Annotation not found or you do not have permission to update it'})
    except Exception as e:
        logger.error(f"Error handling annotation update: {str(e)}")
        emit('error', {'message': 'Error updating annotation'})

@socketio.on('delete_annotation')
def handle_delete_annotation(data):
    try:
        user_id = get_jwt_identity()
        annotation_id = data['id']
        result = db.annotations.delete_one({'_id': ObjectId(annotation_id), 'user_id': user_id})
        if result.deleted_count:
            emit('annotation_deleted', {'id': annotation_id}, room=data['document_id'])
        else:
            emit('error', {'message': 'Annotation not found or you do not have permission to delete it'})
    except Exception as e:
        logger.error(f"Error handling annotation deletion: {str(e)}")
        emit('error', {'message': 'Error deleting annotation'})

if __name__ == '__main__':
    socketio.run(app, debug=True)