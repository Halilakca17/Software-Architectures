# **Software Architectures**

Video Analysis Web Application  
A comprehensive web application that analyzes video files to extract audio emotions, text sentiment, and other key insights using advanced AI models and a Flask-based interface.

---

## **🎯 Project Overview**

This project provides a complete video analysis solution that combines modern web technologies with advanced AI models to derive valuable insights from video content. The application features a user-friendly interface, powerful analysis capabilities, and scalable architecture.

---

## **✨ Key Features**

### **🎥 Video Analysis Pipeline**

- Audio Extraction: Extracts and normalizes audio from video files  
- Speech-to-Text: Utilizes OpenAI Whisper for accurate transcription  
- Text Summarization: Implements T5 model with chunk-based processing  
- Sentiment Analysis: Uses BERTweet model for emotion detection (Joy, Optimism, Anger, Sadness, Neutral)  
- Audio Emotion Analysis: HuBERT model for emotion transition analysis  

### **🌐 Web Interface**

- File Upload: Drag-and-drop support for multiple video formats  
- Real-time Tracking: Live status updates and progress visualization  
- Interactive Results: Charts and detailed analysis reports  
- Responsive Design: Modern UI/UX with Bootstrap integration  

### **🔒 Security & Performance**

- File validation and secure handling  
- Asynchronous processing with threading  
- GPU support for accelerated processing  
- Model caching and batch processing optimization  

---

## 🎯 Pull Request Denemesi

Bu satır, örnek bir pull requesttir.

## **🛠️ Technical Stack**

### Backend

- Flask 2.3.3  
- Python 3.10+  

### Frontend

- HTML5, CSS3  
- JavaScript (Modern ES6+)  

### AI/ML Libraries

- PyTorch 2.0.0+ - Deep learning framework  
- Transformers 4.30.0+ - NLP models  
- OpenAI Whisper 20231117 - Speech-to-text  
- Librosa 0.10.0+ - Audio processing  
- TextBlob 0.17.1+ - Text processing  

### Data Processing

- NumPy 1.26.4+ - Numerical operations  
- Pandas 2.0.0+ - Data analysis  
- Matplotlib 3.7.0+ - Data visualization  
- MoviePy 1.0.3+ - Video processing  

---

## **📁 Project Structure**

- `__pycache__/`           # Python cache files  
- `results/`               # Analysis results storage  
- `templates/`             # HTML templates  
- `uploads/`               # Uploaded video files  
- `app.py`                 # Main Flask application  
- `audio.wav`              # Sample audio file  
- `requirements.txt`       # Python dependencies  
- `setup.sh`               # Setup script  
- `tumkod.py`              # Core analysis modules  
- `simplified_tumkod.py`   # Simplified version  
- `test_pipeline.py`       # Testing utilities  
- various result files      # Analysis outputs  

---

## **🚀 Installation & Setup**

### System Requirements

- Python 3.10+  
- FFmpeg (for video processing)  
- Optional: CUDA-enabled GPU for acceleration  

### Installation Steps

- Clone the repository  
  `git clone https://github.com/Halilakca17/Software-Architectures.git`  
  `cd Software-Architectures`  

- Create virtual environment  
  `python -m venv venv`  
  `source venv/bin/activate`  # On Windows: `venv\Scripts\activate`  

- Install dependencies  
  `pip install -r requirements.txt`  

- Run setup script (if available)  
  `chmod +x setup.sh`  
  `./setup.sh`  

- Launch the application  
  `python app.py`  

- Access the application  
  Open your browser and navigate to [http://localhost:5000](http://localhost:5000)  

---

## **📊 Supported Features**

### File Formats

- MP4, AVI, MOV, MKV, WEBM  
- Maximum file size: 500MB  

### Analysis Capabilities

- Audio Emotions: Real-time emotion detection and visualization  
- Text Sentiment: Multi-category sentiment analysis with confidence scores  
- Speech Transcription: Accurate speech-to-text conversion  
- Content Summarization: Intelligent text summarization  
- Progress Tracking: Real-time analysis status updates  

### Output Formats

- Interactive charts and visualizations  
- Detailed analysis reports  
- Full transcripts  
- Downloadable results  

---

## **🔧 Usage**

- Upload Video: Use the drag-and-drop interface to upload your video file  
- Start Analysis: Click the analyze button to begin processing  
- Monitor Progress: Watch real-time status updates and progress indicators  
- View Results: Explore interactive charts and detailed analysis reports  
- Download Results: Save analysis results for future reference  

---

## **🤖 AI Models Used**

- Whisper: Advanced speech-to-text conversion  
- BERTweet: Social media optimized sentiment analysis  
- HuBERT: Audio emotion recognition  
- T5: Text summarization with fallback strategies  

---

## **🔄 Data Processing Pipeline**

- Video upload and validation  
- Audio extraction and normalization  
- Transcription using Whisper  
- Text summarization with T5  
- Sentiment analysis with BERTweet  
- Audio emotion analysis with HuBERT  
- Results visualization and storage  

---

## **🚧 Future Enhancements**

### Planned Features

- Multi-language support  
- Advanced video analysis  
- User account system  
- API integration  
- Mobile application  
- Real-time analysis  
- Custom model training  

### Areas for Improvement

- Model optimization  
- Enhanced error handling  
- UI/UX improvements  
- Performance tuning  
- Security updates  

---

## **📝 Testing**

Run the test pipeline to verify functionality:  
`python test_pipeline.py`  

---

## **🤝 Contributing**

- Fork the repository  
- Create a feature branch  
- Make your changes  
- Submit a pull request  

---

## **📄 License**

This project is open source. Please refer to the repository for license details.

---
