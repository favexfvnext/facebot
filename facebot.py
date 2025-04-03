import os
import numpy as np
import cv2
from deepface import DeepFace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import logging
import sqlite3
import re
import asyncio
import gradio as gr
from datetime import datetime
from pathlib import Path
import shutil
from telethon import TelegramClient, events
from telethon.tl.types import User, PeerUser
import threading
import time
import queue

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('FaceRecognitionBot')

os.makedirs('faces', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('temp', exist_ok=True)
os.makedirs('profiles', exist_ok=True)
os.makedirs('analyzed_photos', exist_ok=True) 

face_models = {}
face_embeddings = {}

session_file = "my_session"
api_id = YOUR_API_ID 
api_hash = "YOUR_API_HASH"
phone_number = "YOUR_NUMBER"

db_path = "users.db"

SIMILARITY_THRESHOLD = 81.0 

stop_flag = False

active_bots = {}

model_file = 'models/face_models.json'
embeddings_file = 'models/face_embeddings.npy'

if os.path.exists(model_file):
    try:
        with open(model_file, 'r') as f:
            face_models = json.load(f)
            logger.info(f"Loaded {len(face_models)} face models")
    except Exception as e:
        logger.error(f"Error loading models: {e}")

if os.path.exists(embeddings_file):
    try:
        face_embeddings = np.load(embeddings_file, allow_pickle=True).item()
        logger.info(f"Loaded embeddings for {len(face_embeddings)} faces")
        for name, embeddings in face_embeddings.items():
            logger.info(f"  - {name}: {len(embeddings)} embeddings")
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")


def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        city TEXT,
        description TEXT,
        photo_path TEXT,
        similarity REAL,
        reference_model TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS bot_configs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bot_name TEXT UNIQUE,
        reference_model TEXT,
        is_active INTEGER DEFAULT 0
    )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

def insert_into_db(name, age, city, description, photo_path, similarity, reference_model):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO profiles (name, age, city, description, photo_path, similarity, reference_model)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (name, age, city, description, photo_path, similarity, reference_model))
    
    profile_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    logger.info(f"Profile saved to database: {name}, {age}, {city}, similarity: {similarity:.2f}%")
    return profile_id

def get_bot_configs():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, bot_name, reference_model, is_active FROM bot_configs')
    configs = cursor.fetchall()
    
    conn.close()
    return configs

def save_bot_config(bot_name, reference_model, is_active=0):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('SELECT id FROM bot_configs WHERE bot_name = ?', (bot_name,))
    existing = cursor.fetchone()
    
    if existing:
        cursor.execute('''
        UPDATE bot_configs 
        SET reference_model = ?, is_active = ?
        WHERE bot_name = ?
        ''', (reference_model, is_active, bot_name))
    else:
        cursor.execute('''
        INSERT INTO bot_configs (bot_name, reference_model, is_active)
        VALUES (?, ?, ?)
        ''', (bot_name, reference_model, is_active))
    
    conn.commit()
    conn.close()
    logger.info(f"Bot configuration saved: {bot_name} using model {reference_model}")

def update_bot_status(bot_name, is_active):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
    UPDATE bot_configs 
    SET is_active = ?
    WHERE bot_name = ?
    ''', (1 if is_active else 0, bot_name))
    
    conn.commit()
    conn.close()
    logger.info(f"Bot {bot_name} status updated: {'active' if is_active else 'inactive'}")


def parse_bio(bio):
    parts = bio.split(",") 
    if len(parts) < 3: 
        return None, None, None, None 

    name = parts[0].strip() 
    age = re.findall(r'\d+', parts[1])  
    age = age[0] if age else None  
    
    remaining_text = ", ".join(parts[2:]).strip()
    city_desc = remaining_text.split("–", 1)  
    
    city = city_desc[0].strip()
    description = city_desc[1].strip() if len(city_desc) > 1 else ""

    logger.info(f"Parsed profile - Name: {name}, Age: {age}, City: {city}")
    
    return name, age, city, description

def save_models():
    try:
        with open(model_file, 'w') as f:
            json.dump(face_models, f)
        
        np.save(embeddings_file, face_embeddings)
        logger.info(f"Models and embeddings saved successfully")
    except Exception as e:
        logger.error(f"Error saving models: {e}")

def extract_face_embedding(img_path):
    logger.info(f"Extracting embedding from {img_path}")
    try:
        embedding_objs = DeepFace.represent(img_path=img_path, 
                                           model_name="VGG-Face", 
                                           enforce_detection=False,
                                           detector_backend="retinaface")
        
        if embedding_objs and len(embedding_objs) > 0:
            logger.info(f"Successfully extracted embedding from {img_path}")
            return embedding_objs[0]['embedding']
        logger.warning(f"Failed to detect face in {img_path}")
        return None
    except Exception as e:
        logger.error(f"Error extracting embedding from {img_path}: {e}")
        return None

def clean_embedding(embedding):
    if embedding is None:
        return None
    embedding = np.array(embedding)
    embedding = np.nan_to_num(embedding)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding

def train_face(image, name):
    if image is None:
        logger.error("Error: Image not found")
        return "Error: Image not found", None
    
    if not name:
        name = f"Face_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Starting training on one image for '{name}'")
    
    face_dir = f"faces/{name}"
    os.makedirs(face_dir, exist_ok=True)
    
    face_path = f"{face_dir}/main.jpg"
    cv2.imwrite(face_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    logger.info(f"Saved image to {face_path}")
    
    embedding = extract_face_embedding(face_path)
    if embedding is not None:
        embedding = clean_embedding(embedding)
        if embedding is None:
            logger.error(f"Error: Invalid embedding for {face_path}")
            return "Error: Invalid embedding", None
            
        if name not in face_embeddings:
            face_embeddings[name] = []
        face_embeddings[name].append(embedding)
        logger.info(f"Added embedding for '{name}', total embeddings: {len(face_embeddings[name])}")

        face_models[name] = face_dir

        save_models()
        
        return f"Face '{name}' successfully saved!", gr.Dropdown(choices=list(face_models.keys()))
    else:
        logger.error(f"Error: Failed to detect face in the image")
        return "Error: Failed to detect face in the image", None

def train_from_folder(folder_path, name):
    global stop_flag
    
    if not os.path.exists(folder_path):
        logger.error(f"Error: Folder {folder_path} not found")
        return f"Error: Folder {folder_path} not found", None
    
    if not name:
        name = f"Face_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Starting training from folder {folder_path} for '{name}'")
    
    face_dir = f"faces/{name}"
    if os.path.exists(face_dir):
        logger.info(f"Removing existing folder {face_dir}")
        shutil.rmtree(face_dir)
    os.makedirs(face_dir, exist_ok=True)
    image_files = set()
    extensions = ['.jpg', '.jpeg', '.png']

    for ext in extensions:
        for file_path in Path(folder_path).glob(f'*{ext}'):
            image_files.add(str(file_path))
        for file_path in Path(folder_path).glob(f'*{ext.upper()}'):
            image_files.add(str(file_path))
    
    image_files = list(image_files)
    
    logger.info(f"Found {len(image_files)} unique images in folder {folder_path}")
    
    if not image_files:
        logger.error(f"No images found in folder {folder_path}")
        return f"Error: No images found in folder {folder_path}", None
    
    max_images = 50
    if len(image_files) > max_images:
        logger.warning(f"Found too many images ({len(image_files)}). Will only process {max_images}.")
        image_files = image_files[:max_images]
    
    image_count = 0
    embeddings = []
    processed_files = set()
    progress_text = f"Processing 0 of {len(image_files)} images..."
    
    progress_path = "temp_progress.txt"
    with open(progress_path, 'w') as f:
        f.write(progress_text)
    
    try:
        for i, img_path in enumerate(image_files):
            if stop_flag:
                logger.info("Received signal to stop training. Saving intermediate results.")
                break
                
            canonical_path = os.path.normpath(os.path.abspath(img_path))
            if canonical_path in processed_files:
                logger.info(f"Skipping duplicate: {img_path}")
                continue
            
            processed_files.add(canonical_path)
            filename = os.path.basename(img_path)
            logger.info(f"Processing image {i+1}/{len(image_files)}: {filename}")
            
            progress_text = f"Processing {i+1} of {len(image_files)} images...\nCurrent file: {filename}"
            with open(progress_path, 'w') as f:
                f.write(progress_text)
            
            try:
                file_size = os.path.getsize(img_path) / (1024 * 1024)
                if file_size > 10: 
                    logger.warning(f"Skipping {filename}: file size too large ({file_size:.1f}MB)")
                    continue
                
                dst_path = os.path.join(face_dir, filename)
                if os.path.exists(dst_path):
                    base, ext = os.path.splitext(filename)
                    dst_path = os.path.join(face_dir, f"{base}_{i}{ext}")
                
                shutil.copy2(img_path, dst_path)
                logger.info(f"Copied to {dst_path}")
                
                embedding = extract_face_embedding(dst_path)
                if embedding is not None:
                    embedding = clean_embedding(embedding)
                    if embedding is not None:
                        embeddings.append(embedding)
                        image_count += 1
                        logger.info(f"Successfully extracted embedding from {filename}, total: {image_count}")
                    else:
                        logger.warning(f"Invalid embedding for {filename}")
                else:
                    logger.warning(f"Failed to detect face in {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
    
    except Exception as e:
        logger.error(f"Critical error processing folder: {e}")
        if os.path.exists(progress_path):
            os.remove(progress_path)
        return f"Error processing folder: {e}", None
    
    if os.path.exists(progress_path):
        os.remove(progress_path)
    
    stop_flag = False
    
    if image_count > 0:
        face_embeddings[name] = embeddings
        processed_count = len(processed_files)
        
        if processed_count < len(image_files):
            logger.info(f"Training stopped by user. Processed {processed_count} of {len(image_files)} images.")
            status_message = f"STOPPED! Saved {image_count} embeddings for '{name}' from {processed_count} processed images."
        else:
            logger.info(f"Saved {image_count} embeddings for '{name}' from {len(image_files)} images")
            status_message = f"Trained on {image_count} images for '{name}'"
        
        face_models[name] = face_dir
        
        save_models()
        
        return status_message, gr.Dropdown(choices=list(face_models.keys()))
    else:
        logger.error(f"Error: No faces found in folder {folder_path}")
        if os.path.exists(face_dir):
            shutil.rmtree(face_dir)
        return "Error: No faces found", None

def compute_similarity(embedding1, embedding2):
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    dot_product = np.dot(embedding1, embedding2)
    similarity = dot_product / (norm1 * norm2)
    
    if similarity < 0.5:
        similarity_pct = max(0, similarity * 40)
    else:
        similarity_pct = 20 + (similarity - 0.5) * 160
    
    return max(0, min(100, similarity_pct))

def compare_face(img_path, reference_face):
    if not os.path.exists(img_path):
        logger.error(f"Error: Image {img_path} not found")
        return None
    
    if reference_face not in face_models or reference_face not in face_embeddings:
        logger.error(f"Error: Reference face '{reference_face}' not found or not trained")
        return None
    
    reference_embeddings = face_embeddings[reference_face]
    if not reference_embeddings:
        logger.error(f"Error: No saved embeddings for reference face '{reference_face}'")
        return None
    
    logger.info(f"Starting comparison with reference face '{reference_face}' ({len(reference_embeddings)} embeddings)")
    
    try:
        test_embedding = extract_face_embedding(img_path)
        if test_embedding is None:
            logger.error(f"Failed to detect face in {img_path}")
            return None
        
        test_embedding = clean_embedding(test_embedding)
        if test_embedding is None:
            logger.error(f"Invalid embedding for {img_path}")
            return None
        
        similarities = [compute_similarity(test_embedding, ref_emb) for ref_emb in reference_embeddings]
        max_similarity = max(similarities)
        
        logger.info(f"Similarity for {img_path}: {max_similarity:.2f}%")
        return max_similarity
    
    except Exception as e:
        logger.error(f"Error comparing faces: {e}")
        return None

def delete_face_model(name):
    if name not in face_models:
        logger.error(f"Model {name} not found")
        return f"Error: Model {name} not found"
    
    try:
        face_dir = face_models[name]
        del face_models[name]
        if name in face_embeddings:
            del face_embeddings[name]
        
        if os.path.exists(face_dir):
            shutil.rmtree(face_dir)
            
        save_models()
        
        logger.info(f"Model {name} successfully deleted")
        return f"Model {name} successfully deleted", gr.Dropdown(choices=list(face_models.keys()))
    except Exception as e:
        logger.error(f"Error deleting model {name}: {e}")
        return f"Error deleting model {name}: {e}", None

def stop_training():
    global stop_flag
    stop_flag = True
    logger.info("Training stop requested")
    return "Stopping training... Please wait. Intermediate results will be saved."


latest_matches = []
MAX_LATEST_MATCHES = 20

chart_update_queue = queue.Queue()

def add_match_record(name, similarity, matched, reference_face):
    """Add a record of a match attempt to the recent matches list"""
    global latest_matches
    
    latest_matches.insert(0, {
        'name': name,
        'similarity': similarity,
        'matched': matched,
        'reference_face': reference_face,
        'timestamp': datetime.now().strftime('%H:%M:%S')
    })
    
    if len(latest_matches) > MAX_LATEST_MATCHES:
        latest_matches = latest_matches[:MAX_LATEST_MATCHES]
    
    logger.info(f"Added match record: {name}, similarity: {similarity:.2f}%, matched: {matched}")
    
    try:
        chart_update_queue.put_nowait(True)
    except:
        pass

def generate_matches_chart():
    try:
        if not latest_matches:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "No matches data yet", ha='center', va='center', fontsize=14)
            plt.axis('off')
            chart_path = "temp/matches_chart.png"
            plt.savefig(chart_path)
            plt.close('all')
            return chart_path
            
        with threading.Lock():
            matches_copy = latest_matches.copy()
        
        names = [m['name'] if len(m['name']) <= 15 else m['name'][:12]+'...' for m in matches_copy]
        similarities = [m['similarity'] for m in matches_copy]
        colors = ['green' if m['matched'] else 'red' for m in matches_copy]
        
        plt.figure(figsize=(12, 7))
        bars = plt.barh(range(len(similarities)), similarities, color=colors)
        
        for i, (bar, value, name) in enumerate(zip(bars, similarities, names)):
            plt.text(value + 1, i, f"{value:.1f}%", va='center', fontsize=10)
            
        plt.yticks(range(len(names)), names)
        plt.xlabel('Similarity (%)')
        plt.title('Recent Profile Comparisons')
        
        plt.axvline(x=SIMILARITY_THRESHOLD, color='blue', linestyle='--', 
                    label=f'Threshold ({SIMILARITY_THRESHOLD}%)')
        plt.legend()
        
        plt.figtext(0.98, 0.02, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                    ha='right', fontsize=8)
        
        chart_path = "temp/matches_chart.png"
        plt.savefig(chart_path, dpi=100, bbox_inches='tight')
        plt.close('all') 
        
        return chart_path
    except Exception as e:
        logger.error(f"Error generating chart: {e}")
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error generating chart: {e}", ha='center', va='center', fontsize=12, color='red')
        plt.axis('off')
        chart_path = "temp/matches_chart_error.png"
        plt.savefig(chart_path)
        plt.close('all')
        return chart_path

async def analyze_profile(event, client, reference_face):
    try:
        sender = await event.get_sender()
        if not isinstance(sender, User):
            return
        
        message = event.message
        if not message or not message.message:
            logger.warning("No message content to parse")
            return
        
        if hasattr(message, 'video') and message.video:
            logger.info(f"Message from {sender.id} contains video, sending rejection code")
            await client.send_message(sender, "3")
            return
        
        name, age, city, description = parse_bio(message.message)
        if not name or not age or not city:
            logger.warning(f"Could not parse complete profile information: {message.message}")
            return
        
        unique_id = f"{sender.id}_{int(time.time())}"
        photo_path = f"temp/{unique_id}.jpg"
        photo_found = False

        if message.media and hasattr(message.media, 'photo'):
            logger.info(f"Found photo in message from {name}")
            await client.download_media(message.media, photo_path)
            photo_found = True
        
        if not photo_found:
            logger.info(f"No photo in message from {name}, checking profile photos")
            profile_photos = await client.get_profile_photos(sender)
            if profile_photos:
                logger.info(f"Using profile photo for {name}")
                await client.download_media(profile_photos[0], photo_path)
                photo_found = True
            else:
                logger.warning(f"No photos found for {name}")
                await client.send_message(sender, "3")
                return
        
        if not photo_found or not os.path.exists(photo_path):
            logger.warning(f"Failed to download photo for {name}")
            await client.send_message(sender, "3")
            return
        
        similarity = compare_face(photo_path, reference_face)
        
        if similarity is None:
            logger.warning(f"Could not compare face for {name}")
            await client.send_message(sender, "3")
            os.remove(photo_path)
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(add_match_record, name, 0, False, reference_face)
            return
        
        logger.info(f"Profile analysis - Name: {name}, Age: {age}, City: {city}, Similarity: {similarity:.2f}%")
        matched = similarity >= SIMILARITY_THRESHOLD

        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(add_match_record, name, similarity, matched, reference_face)
        
        if matched:
            saved_photo_path = f"analyzed_photos/{unique_id}.jpg"
            shutil.copy2(photo_path, saved_photo_path)
            
            insert_into_db(name, age, city, description, saved_photo_path, similarity, reference_face)
            
            await client.send_message(sender, "1")
            logger.info(f"Match found: {name}, {age}, {city} - Similarity: {similarity:.2f}%")
        else:
            await client.send_message(sender, "3")
            logger.info(f"Rejected: {name}, {age}, {city} - Similarity: {similarity:.2f}%")
        
        os.remove(photo_path)
        
    except Exception as e:
        logger.error(f"Error analyzing profile: {e}")
        try:
            await client.send_message(sender, "3")
        except:
            pass

async def start_bot(bot_name, reference_face):
    logger.info(f"Starting bot '{bot_name}' with reference face '{reference_face}'")
    
    os.makedirs('sessions', exist_ok=True)
    
    client = TelegramClient(f"sessions/{bot_name}", api_id, api_hash)
    await client.start(phone_number)
    
    update_bot_status(bot_name, True)
    
    active_bots[bot_name] = {
        'client': client,
        'reference_face': reference_face,
        'start_time': datetime.now()
    }
    
    @client.on(events.NewMessage)
    async def handle_message(event):
        if event.is_private:
            message = event.message
            logger.info(f"Received message from {event.sender_id}: {message.message[:30]}...")
            if message.media:
                media_type = "Unknown"
                if hasattr(message.media, 'photo'):
                    media_type = "Photo"
                elif hasattr(message.media, 'video'):
                    media_type = "Video"
                logger.info(f"Message has media: {media_type}")
            
            asyncio.create_task(analyze_profile(event, client, reference_face))
    
    try:
        logger.info(f"Bot '{bot_name}' is now running")
        await client.run_until_disconnected()
    except Exception as e:
        logger.error(f"Bot '{bot_name}' error: {e}")
    finally:
        update_bot_status(bot_name, False)
        if bot_name in active_bots:
            del active_bots[bot_name]
        logger.info(f"Bot '{bot_name}' stopped")

async def stop_bot(bot_name):
    if bot_name in active_bots:
        logger.info(f"Stopping bot '{bot_name}'")
        try:
            await active_bots[bot_name]['client'].disconnect()
            update_bot_status(bot_name, False)
            del active_bots[bot_name]
            return f"Bot '{bot_name}' stopped successfully"
        except Exception as e:
            logger.error(f"Error stopping bot '{bot_name}': {e}")
            return f"Error stopping bot: {e}"
    else:
        return f"Bot '{bot_name}' is not running"

def start_bot_thread(bot_name, reference_face):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_bot(bot_name, reference_face))

def update_reference_dropdown():
    return gr.Dropdown(choices=list(face_models.keys()))

def update_bot_list():
    configs = get_bot_configs()
    bot_list = [f"{cfg[1]} - {cfg[2]} - {'Active' if cfg[3] else 'Inactive'}" for cfg in configs]
    return gr.Dropdown(choices=bot_list)

def launch_bot(bot_name, reference_face):
    if not bot_name or not reference_face:
        return "Error: Please provide a bot name and select a reference face"
    
    if bot_name in active_bots:
        return f"Error: Bot '{bot_name}' is already running"
    
    save_bot_config(bot_name, reference_face, 1)
    
    os.makedirs('sessions', exist_ok=True)
    
    thread = threading.Thread(target=start_bot_thread, args=(bot_name, reference_face))
    thread.daemon = True
    thread.start()
    
    return f"Bot '{bot_name}' started with reference face '{reference_face}'"

def stop_running_bot(bot_selection):
    if not bot_selection:
        return "Error: Please select a bot to stop"
    
    bot_name = bot_selection.split(" - ")[0]
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(stop_bot(bot_name))
    
    return result

def update_statistics_chart():
    try:
        chart_path = generate_matches_chart()
        return chart_path
    except Exception as e:
        logger.error(f"Error updating statistics chart: {e}")
        return None


def create_interface():
    with gr.Blocks(title="Face Recognition Bot System") as app:
        gr.Markdown("# Face Recognition Bot System")
        
        with gr.Tab("Train Face Models"):
            gr.Markdown("## Train the system on reference faces")
            
            with gr.Group():
                gr.Markdown("### Method 1: Train on a single image")
                with gr.Row():
                    train_image = gr.Image(label="Upload a photo for training")
                    train_name = gr.Textbox(label="Name/label for the face")
                train_button = gr.Button("Train", variant="primary")
            
            with gr.Group():
                gr.Markdown("### Method 2: Train on a folder of images (recommended)")
                with gr.Row():
                    folder_path_train = gr.Textbox(label="Path to folder with face images")
                    train_name_folder = gr.Textbox(label="Name/label for the face")
                
                with gr.Row():
                    train_folder_button = gr.Button("Train on Folder", variant="primary")
                    stop_button = gr.Button("Emergency Stop", variant="stop")
                
                train_progress = gr.Textbox(label="Training progress", visible=True)
            
            with gr.Group():
                gr.Markdown("### Manage Models")
                with gr.Row():
                    delete_model_dropdown = gr.Dropdown(choices=list(face_models.keys()), 
                                                       label="Select model to delete")
                    delete_button = gr.Button("Delete Model", variant="stop")
            
            train_output = gr.Textbox(label="Result")
        
        with gr.Tab("Manage Bots"):
            gr.Markdown("## Configure and manage Telegram bots")
            
            with gr.Group():
                gr.Markdown("### Start a new bot")
                with gr.Row():
                    bot_name = gr.Textbox(label="Bot name (unique identifier)")
                    reference_face_dropdown = gr.Dropdown(choices=list(face_models.keys()), 
                                                         label="Select reference face model")
                start_bot_button = gr.Button("Start Bot", variant="primary")
                bot_status = gr.Textbox(label="Bot status")
            
            with gr.Group():
                gr.Markdown("### Manage running bots")
                with gr.Row():
                    bot_list_dropdown = gr.Dropdown(label="Select a bot", 
                                                   choices=[])
                    refresh_bot_list = gr.Button("Refresh List")
                
                stop_bot_button = gr.Button("Stop Selected Bot", variant="stop")
                bot_action_status = gr.Textbox(label="Action status")
        
        with gr.Tab("Monitoring"):
            gr.Markdown("## Real-time monitoring of bot activity")
            
            with gr.Group():
                gr.Markdown("### Recent matches statistics")
                stats_chart = gr.Image(label="Recent profile comparisons", value=generate_matches_chart())
                refresh_stats_button = gr.Button("Refresh Statistics", variant="primary")
                
                gr.Markdown("### Auto-refresh")
                with gr.Row():
                    auto_refresh = gr.Checkbox(label="Enable auto-refresh", value=False)
                    refresh_interval = gr.Slider(minimum=5, maximum=60, value=10, step=5, 
                                               label="Refresh interval (seconds)")
        
        with gr.Tab("Settings"):
            gr.Markdown("## Configure system settings")
            
            with gr.Group():
                gr.Markdown("### Matching threshold")
                threshold_slider = gr.Slider(minimum=60, maximum=95, value=SIMILARITY_THRESHOLD, 
                                            step=1, label="Similarity threshold (%)")
                save_threshold_button = gr.Button("Save Threshold", variant="primary")
                threshold_status = gr.Textbox(label="Threshold status")
        
        train_button.click(train_face, 
                           inputs=[train_image, train_name], 
                           outputs=[train_output, reference_face_dropdown])
        
        stop_button.click(stop_training, 
                          inputs=[], 
                          outputs=[train_progress])
        
        def folder_training_with_progress(folder_path, name):
            global stop_flag
            stop_flag = False
            
            with open("temp_progress.txt", 'w') as f:
                f.write("Starting processing...")
            
    
            progress_update_interval = 0.5 
            
            def update_progress():
                if os.path.exists("temp_progress.txt"):
                    with open("temp_progress.txt", 'r') as f:
                        return f.read()
                return "Finishing..."
            
            result = [None, None]
            
            def train_thread():
                nonlocal result
                result = train_from_folder(folder_path, name)
            
            thread = threading.Thread(target=train_thread)
            thread.start()
            
            start_time = time.time()
            timeout = 300 
            
            while thread.is_alive():
                progress = update_progress()
                yield progress, None, None
                
                if time.time() - start_time > timeout:
                    logger.error("Timeout exceeded for folder training")
                    stop_flag = True
                    return "Error: Timeout exceeded. Check folder path and contents.", None, None
                
                time.sleep(progress_update_interval)
            

            return result[0], result[1], "Done!"
        
        train_folder_button.click(
            folder_training_with_progress, 
            inputs=[folder_path_train, train_name_folder],
            outputs=[train_output, reference_face_dropdown, train_progress]
        )
        
        delete_button.click(delete_face_model, 
                           inputs=[delete_model_dropdown], 
                           outputs=[train_output, delete_model_dropdown])
        

        refresh_stats_button.click(update_statistics_chart, inputs=[], outputs=[stats_chart])
        

        def toggle_auto_refresh(enabled, interval):
            if enabled:
                return f"Auto-refresh enabled, will update every {interval} seconds"
            else:
                return "Auto-refresh disabled"
        
        auto_refresh.change(toggle_auto_refresh, inputs=[auto_refresh, refresh_interval], outputs=[bot_action_status])
        

        def chart_updater():
            while True:
                try:

                    try:
                        chart_update_queue.get(timeout=5)

                        if auto_refresh.value:
                            chart_path = update_statistics_chart()
                            if chart_path:
                                stats_chart.update(value=chart_path)
                    except queue.Empty:

                        if auto_refresh.value:                       
                            current_time = time.time()
                            if not hasattr(chart_updater, 'last_update') or current_time - chart_updater.last_update >= refresh_interval.value:
                                chart_path = update_statistics_chart()
                                if chart_path:
                                    stats_chart.update(value=chart_path)
                                chart_updater.last_update = current_time
                except Exception as e:
                    logger.error(f"Error in chart updater: {e}")
                    time.sleep(10)
                
                time.sleep(1) 
        
        chart_thread = threading.Thread(target=chart_updater, daemon=True)
        chart_thread.start()
        
        start_bot_button.click(launch_bot,
                              inputs=[bot_name, reference_face_dropdown],
                              outputs=[bot_status])
        
        refresh_bot_list.click(update_bot_list,
                              inputs=[],
                              outputs=[bot_list_dropdown])
        
        stop_bot_button.click(stop_running_bot,
                             inputs=[bot_list_dropdown],
                             outputs=[bot_action_status])
        
        def save_threshold(value):
            global SIMILARITY_THRESHOLD
            SIMILARITY_THRESHOLD = value
            logger.info(f"updated to {value}%")
            return f"updated to {value}%"
        
        save_threshold_button.click(save_threshold,
                                   inputs=[threshold_slider],
                                   outputs=[threshold_status])
        
        refresh_bot_list.click(fn=None)
    
    return app


if __name__ == "__main__":
    try:
        init_db()
        
        app = create_interface()
        
        app.launch()
    except Exception as e:
        logger.error(e) 