import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import threading
from tumkod import MediaAnalysisPipeline

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['RESULTS_FOLDER'], 'images'), exist_ok=True)

# Dictionary to store analysis status
analysis_tasks = {}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov', 'mkv', 'webm'}

def run_analysis(video_path, task_id):
    """Run the media analysis pipeline in a separate thread"""
    try:
        # Update task status
        analysis_tasks[task_id]['status'] = 'processing'
        
        # Create and run the pipeline
        pipeline = MediaAnalysisPipeline()
        
        # Override default file paths to use our task_id
        output_base = os.path.join(app.config['RESULTS_FOLDER'], task_id)
        pipeline.audio_path = os.path.join(output_base, "audio.wav")
        pipeline.transcript_path = os.path.join(output_base, "transcript.txt")
        pipeline.summary_path = os.path.join(output_base, "summary.txt")
        pipeline.sentiment_results_path = os.path.join(output_base, "sentiment_results.txt")
        pipeline.sentiment_chart_path = os.path.join(output_base, "sentiment_chart.png")
        
        # Make sure the output directory exists
        os.makedirs(output_base, exist_ok=True)
        
        # Run the pipeline
        result = pipeline.run_full_pipeline(video_path, visualize=True)
        
        # Move the emotion chart if it was created
        try:
            emotion_chart = "audio_emotion_analysis.png"
            if os.path.exists(emotion_chart):
                os.rename(emotion_chart, os.path.join(output_base, "audio_emotion_analysis.png"))
        except Exception as e:
            print(f"Error moving emotion chart: {str(e)}")
        
        # Update task status
        if result:
            analysis_tasks[task_id]['status'] = 'completed'
            analysis_tasks[task_id]['results'] = {
                'transcript': pipeline.transcript_path,
                'summary': pipeline.summary_path,
                'sentiment_results': pipeline.sentiment_results_path,
                'sentiment_chart': pipeline.sentiment_chart_path,
                'audio_emotion': result.get('audio_emotion', None)
            }
        else:
            analysis_tasks[task_id]['status'] = 'failed'
            analysis_tasks[task_id]['error'] = 'Pipeline processing failed'
    
    except Exception as e:
        analysis_tasks[task_id]['status'] = 'failed'
        analysis_tasks[task_id]['error'] = str(e)
        print(f"Error in analysis thread: {str(e)}")

@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start analysis"""
    if 'video' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['video']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Generate a unique ID for this analysis task
        task_id = str(uuid.uuid4())
        
        # Create a directory for this task
        task_dir = os.path.join(app.config['RESULTS_FOLDER'], task_id)
        os.makedirs(task_dir, exist_ok=True)
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
        file.save(file_path)
        
        # Initialize the task in our dictionary
        analysis_tasks[task_id] = {
            'status': 'starting',
            'filename': filename,
            'file_path': file_path
        }
        
        # Start analysis in a background thread
        thread = threading.Thread(target=run_analysis, args=(file_path, task_id))
        thread.daemon = True
        thread.start()
        
        # Redirect to status page
        return redirect(url_for('status', task_id=task_id))
    
    flash('Invalid file format. Allowed formats: mp4, avi, mov, mkv, webm')
    return redirect(url_for('index'))

@app.route('/status/<task_id>')
def status(task_id):
    """Show analysis status and results"""
    if task_id not in analysis_tasks:
        flash('Analysis task not found')
        return redirect(url_for('index'))
    
    return render_template('status.html', task_id=task_id, task=analysis_tasks[task_id])

@app.route('/api/status/<task_id>')
def get_status(task_id):
    """API endpoint for getting the status of an analysis task"""
    if task_id not in analysis_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    return jsonify(analysis_tasks[task_id])

@app.route('/results/<task_id>/<path:filename>')
def result_file(task_id, filename):
    """Serve result files"""
    return send_from_directory(os.path.join(app.config['RESULTS_FOLDER'], task_id), filename)

@app.route('/file_content/<task_id>/<path:filename>')
def file_content(task_id, filename):
    """Return text file content as JSON"""
    try:
        # Basit dosya yolu oluştur - tam yolu değil
        file_path = os.path.join(app.config['RESULTS_FOLDER'], task_id, filename)
        
        # Dosya yolu hata ayıklama için yazdır
        print(f"Accessing file at: {file_path}")
        
        # Dosya var mı kontrol et
        if not os.path.exists(file_path):
            # Eğer varsa results/{task_id} içeren yolu temizle
            if f'results/{task_id}' in filename:
                clean_filename = filename.replace(f'results/{task_id}/', '')
                file_path = os.path.join(app.config['RESULTS_FOLDER'], task_id, clean_filename)
                print(f"Path corrected to: {file_path}")
        
        # Dosyayı aç ve içeriği oku
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({'content': content})
    except Exception as e:
        print(f"Error in file_content: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)