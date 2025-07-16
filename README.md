🎯 Project Overview<br>
This project provides a complete video analysis solution that combines modern web technologies with advanced AI models to derive valuable insights from video content. The application features a user-friendly interface, powerful analysis capabilities, and scalable architecture.<br>

✨ Key Features<br>
🎥 Video Analysis Pipeline<br>

Audio Extraction: Extracts and normalizes audio from video files<br>

Speech-to-Text: Utilizes OpenAI Whisper for accurate transcription<br>

Text Summarization: Implements T5 model with chunk-based processing<br>

Sentiment Analysis: Uses BERTweet model for emotion detection (Joy, Optimism, Anger, Sadness, Neutral)<br>

Audio Emotion Analysis: HuBERT model for emotion transition analysis<br>

🌐 Web Interface<br>

File Upload: Drag-and-drop support for multiple video formats<br>

Real-time Tracking: Live status updates and progress visualization<br>

Interactive Results: Charts and detailed analysis reports<br>

Responsive Design: Modern UI/UX with Bootstrap integration<br>

🔒 Security & Performance<br>

File validation and secure handling<br>

Asynchronous processing with threading<br>

GPU support for accelerated processing<br>

Model caching and batch processing optimization<br>

🛠️ Technical Stack<br>
Backend<br>

Flask 2.3.3<br>

Python 3.10+<br>

Frontend<br>

HTML5<br>

CSS3<br>

JavaScript (Modern ES6+)<br>

AI/ML Libraries<br>

PyTorch 2.0.0+ - Deep learning framework<br>

Transformers 4.30.0+ - NLP models<br>

OpenAI Whisper 20231117 - Speech-to-text<br>

Librosa 0.10.0+ - Audio processing<br>

TextBlob 0.17.1+ - Text processing<br>

Data Processing<br>

NumPy 1.26.4+ - Numerical operations<br>

Pandas 2.0.0+ - Data analysis<br>

Matplotlib 3.7.0+ - Data visualization<br>

MoviePy 1.0.3+ - Video processing<br>

📁 Project Structure<br>
├── pycache/ # Python cache files<br>
├── results/ # Analysis results storage<br>
├── templates/ # HTML templates<br>
├── uploads/ # Uploaded video files<br>
├── app.py # Main Flask application<br>
├── audio.wav # Sample audio file<br>
├── requirements.txt # Python dependencies<br>
├── setup.sh # Setup script<br>
├── tumkod.py # Core analysis modules<br>
├── simplified_tumkod.py # Simplified version<br>
├── test_pipeline.py # Testing utilities<br>
└── various result files # Analysis outputs<br>

🚀 Installation & Setup<br>
System Requirements<br>

Python 3.10+<br>

FFmpeg (for video processing)<br>

Optional: CUDA-enabled GPU for acceleration<br>

Installation Steps<br>
1️⃣ Clone the repository<br>

bash<br>
Kopyala
Düzenle
git clone https://github.com/Halilakca17/Software-Architectures.git<br>
cd Software-Architectures<br>
```<br>
2️⃣ Create virtual environment<br>
```bash<br>
python -m venv venv<br>
source venv/bin/activate  # On Windows: venv\Scripts\activate<br>
```<br>
3️⃣ Install dependencies<br>
```bash<br>
pip install -r requirements.txt<br>
```<br>
4️⃣ Run setup script (if available)<br>
```bash<br>
chmod +x setup.sh<br>
./setup.sh<br>
```<br>
5️⃣ Launch the application<br>
```bash<br>
python app.py<br>
```<br>
6️⃣ Access the application<br>
Open your browser and navigate to http://localhost:5000<br>

📊 **Supported Features**<br>
**File Formats**<br>
- Video: MP4, AVI, MOV, MKV, WEBM<br>
- Maximum file size: 500MB<br>

**Analysis Capabilities**<br>
- Audio Emotions: Real-time emotion detection and visualization<br>
- Text Sentiment: Multi-category sentiment analysis with confidence scores<br>
- Speech Transcription: Accurate speech-to-text conversion<br>
- Content Summarization: Intelligent text summarization<br>
- Progress Tracking: Real-time analysis status updates<br>

**Output Formats**<br>
- Interactive charts and visualizations<br>
- Detailed analysis reports<br>
- Full transcripts<br>
- Downloadable results<br>

🔧 **Usage**<br>
- Upload Video: Use the drag-and-drop interface to upload your video file<br>
- Start Analysis: Click the analyze button to begin processing<br>
- Monitor Progress: Watch real-time status updates and progress indicators<br>
- View Results: Explore interactive charts and detailed analysis reports<br>
- Download Results: Save analysis results for future reference<br>

🤖 **AI Models Used**<br>
**Whisper**<br>
- Purpose: Speech-to-Text<br>
- Description: Advanced speech-to-text conversion<br>

**BERTweet**<br>
- Purpose: Sentiment Analysis<br>
- Description: Social media optimized sentiment analysis<br>

**HuBERT**<br>
- Purpose: Audio Emotion<br>
- Description: Audio emotion recognition<br>

**T5**<br>
- Purpose: Text Summarization<br>
- Description: Text summarization with fallback strategies<br>

🔄 **Data Processing Pipeline**<br>
```mermaid<br>
graph TD<br>
    A[Video Upload] --> B[Validation]<br>
    B --> C[Audio Extraction]<br>
    C --> D[Normalization]<br>
    D --> E[Transcription - Whisper]<br>
    E --> F[Text Summarization - T5]<br>
    F --> G[Sentiment Analysis - BERTweet]<br>
    G --> H[Audio Emotion Analysis - HuBERT]<br>
    H --> I[Results Visualization]<br>
    I --> J[Storage]<br>
```<br>

🚧 **Future Enhancements**<br>
**Planned Features**<br>
- Multi-language support<br>
- Advanced video analysis<br>
- User account system<br>
- API integration<br>
- Mobile application<br>
- Real-time analysis<br>
- Custom model training<br>

**Areas for Improvement**<br>
- Model optimization<br>
- Enhanced error handling<br>
- UI/UX improvements<br>
- Performance tuning<br>
- Security updates<br>

📝 **Testing**<br>
Run the test pipeline to verify functionality:<br>
```bash<br>
python test_pipeline.py<br>
```<br>

🤝 **Contributing**<br>
We welcome contributions! Please follow these steps:<br>
1️⃣ Fork the repository<br>
2️⃣ Create a feature branch (git checkout -b feature/amazing-feature)<br>
3️⃣ Make your changes<br>
4️⃣ Commit your changes (git commit -m 'Add some amazing feature')<br>
5️⃣ Push to the branch (git push origin feature/amazing-feature)<br>
6️⃣ Submit a pull request<br>

📄 **License**<br>
This project is open source. Please refer to the repository for license details.<br>

📞 **Support**<br>
For issues and questions, please visit the GitHub repository and create an issue.<br>

🌟 **Acknowledgments**<br>
- OpenAI for the Whisper model<br>
- Hugging Face for the Transformers library<br>
- The PyTorch team for the deep learning framework<br>
- All contributors and supporters of this project<br>

---
