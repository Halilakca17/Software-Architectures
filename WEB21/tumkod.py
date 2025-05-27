import os
import re
import warnings
import matplotlib.pyplot as plt
from collections import Counter
from moviepy import VideoFileClip
import numpy as np
import torch
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
from pydub import AudioSegment
from scipy.io import wavfile
from textblob import TextBlob  # As a fallback for sentiment analysis

# Check and set environment paths
if os.name == 'nt':  # Windows
    os.environ["PATH"] += os.pathsep + r"C:\ffmpeg"

# Suppress warnings
warnings.filterwarnings("ignore")

class MediaAnalysisPipeline:
    """
    A complete pipeline for media analysis:
    1. Extract audio from video
    2. Transcribe audio to text
    3. Summarize text
    4. Analyze text sentiment
    5. Analyze audio emotions
    """
    
    def __init__(self):
        """Initialize the pipeline with all necessary components"""
        print("Initializing Media Analysis Pipeline...")
        
        # Paths
        self.video_path = None
        self.audio_path = None
        self.transcript_path = "toplanti_metni2.txt"
        self.summary_path = "toplanti_ozeti2.txt"
        self.sentiment_results_path = "duygu_analizi_sonuclari2.txt"
        self.sentiment_chart_path = "duygu_dagilimi2.png"
        
        # Models and processors will be loaded on demand
        self.whisper_model = None
        self.summarizer = None
        self.sentiment_analyzer = None
        self.audio_emotion_analyzer = None
        
        print("Pipeline initialized and ready to use.")
        
    def extract_audio(self, video_path, audio_path="audio.wav"):
        """Extract audio from video file"""
        print(f"Extracting audio from {video_path}...")
        try:
            self.video_path = video_path
            self.audio_path = audio_path
            
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path)
            
            print(f"Audio successfully saved to: {audio_path}")
            return True
        except Exception as e:
            print(f"Error extracting audio: {str(e)}")
            return False
    
    def transcribe_audio(self, audio_path=None):
        """Transcribe audio to text using Whisper"""
        if audio_path is None:
            audio_path = self.audio_path
            
        print(f"Transcribing audio from {audio_path}...")
        try:
            # Import whisper here to avoid early import issues
            import whisper
            
            if self.whisper_model is None:
                print("Loading Whisper model...")
                self.whisper_model = whisper.load_model("base")
                
            result = self.whisper_model.transcribe(audio_path)
            
            with open(self.transcript_path, "w", encoding="utf-8") as f:
                f.write(result['text'])
                
            print(f"Transcript saved to {self.transcript_path}")
            return result['text']
        except ImportError:
            print("Whisper package not found. Please install with: pip install openai-whisper")
            return None
        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            return None
    
    def summarize_text(self, text=None, max_chunk_length=1000):
        """Summarize text using a transformers pipeline"""
        if text is None:
            try:
                with open(self.transcript_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                print(f"Error reading transcript: {str(e)}")
                return None
        
        print("Summarizing text...")
        try:
            
            from transformers import pipeline
            
            if self.summarizer is None:
                print("Loading summarization model...")
                self.summarizer = pipeline("summarization", model="t5-small")
            
        
            chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
            
           
            summary = " ".join([
                self.summarizer(chunk, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
                for chunk in chunks
            ])
            
            # Save summary
            with open(self.summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
            
            print(f"Summary saved to {self.summary_path}")
            return summary
        except Exception as e:
            print(f"Error summarizing text: {str(e)}")
            print("Attempting fallback summarization...")
            
            # Fallback simple summarization (extract first few sentences)
            sentences = re.split(r'(?<=[.!?])\s+', text)
            simple_summary = ' '.join(sentences[:min(5, len(sentences))])
            
            with open(self.summary_path, "w", encoding="utf-8") as f:
                f.write(simple_summary)
                
            print(f"Simple summary saved to {self.summary_path}")
            return simple_summary
    """
    def analyze_text_sentiment(self, text=None):
        Analyze sentiment in text
        if text is None:
            try:
                with open(self.transcript_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                print(f"Error reading transcript: {str(e)}")
                return None
        
        print("Analyzing text sentiment...")
        
        # Split text into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        
        try:
            # Import transformers pipeline
            from transformers import pipeline
            
            if self.sentiment_analyzer is None:
                print("Loading sentiment analysis model...")
                # Yeni model - siebert/sentiment-roberta-large-english
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis", 
                    model="siebert/sentiment-roberta-large-english",
                    device=-1  # Force CPU usage (GPU için 0 kullanın)
                )
            
            # Daha verimli batch işleme
            batch_size = 8
            results = []
            
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                batch_results = self.sentiment_analyzer(batch)
                results.extend(batch_results)
                
            # Bu model pozitif/negatif sonuç verir, 
            # bunu diğer duygusal sınıflara dönüştürelim
            emotion_results = []
            for res in results:
                label = res['label']
                score = res['score']
                
                # Label dönüşümü (POSITIVE/NEGATIVE -> duygu etiketleri)
                if label == "POSITIVE":
                    if score > 0.9:
                        emotion = "joy"
                    else:
                        emotion = "optimism"
                else:  # NEGATIVE
                    if score > 0.9:
                        emotion = "anger"
                    else:
                        emotion = "sadness"
                        
                emotion_results.append({"label": emotion, "score": score})
                
        except Exception as e:
            print(f"Error using transformers for sentiment analysis: {str(e)}")
            print("Falling back to TextBlob for sentiment analysis...")
            
            # Fallback to TextBlob
            emotion_results = []
            for sentence in sentences:
                analysis = TextBlob(sentence)
                
                # Map polarity to emotions (simplified)
                if analysis.polarity > 0.5:
                    emotion = "joy"
                elif analysis.polarity > 0.1:
                    emotion = "optimism"
                elif analysis.polarity < -0.5:
                    emotion = "anger"
                elif analysis.polarity < -0.1:
                    emotion = "sadness"
                else:
                    emotion = "neutral"
                
                emotion_results.append({"label": emotion, "score": abs(analysis.polarity)})
        
        # Save results to file
        with open(self.sentiment_results_path, "w", encoding="utf-8") as f_out:
            for sentence, res in zip(sentences, emotion_results):
                output = f"Cümle: {sentence}\nDuygu: {res['label']}, Güven: {res['score']:.2f}\n\n"
                f_out.write(output)
        
        # Create visualization
        labels = [res['label'] for res in emotion_results]
        label_counts = Counter(labels)
        
        plt.figure(figsize=(6, 6))
        plt.pie(
            label_counts.values(), 
            labels=label_counts.keys(), 
            autopct='%1.1f%%', 
            colors=['blue', 'red', 'orange', 'green', 'purple', 'pink']
        )
        plt.title("Toplantı Duygu Dağılımı")
        plt.savefig(self.sentiment_chart_path)
        
        print(f"Sentiment analysis results saved to {self.sentiment_results_path}")
        print(f"Sentiment chart saved to {self.sentiment_chart_path}")
        
        return emotion_results
    """
    
    def analyze_text_sentiment(self, text=None):
        """Analyze sentiment in text"""
        if text is None:
            try:
                with open(self.transcript_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                print(f"Error reading transcript: {str(e)}")
                return None
        
        print("Analyzing text sentiment...")
        
        # Split text into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        
        try:
            # Import transformers pipeline
            from transformers import pipeline
            
            if self.sentiment_analyzer is None:
                print("Loading sentiment analysis model...")
                # BERTweet modeli ile duygu analizi
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis", 
                    model="finiteautomata/bertweet-base-sentiment-analysis",
                    device=-1  # CPU için -1, GPU için 0
                )
            
            # Daha verimli batch işleme
            batch_size = 8
            results = []
            
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                batch_results = self.sentiment_analyzer(batch)
                results.extend(batch_results)
                
            # Bu model 3 kategorili duygu sonucu verir: POS, NEG, NEU
            # Bu etiketleri daha ayrıntılı duygusal sınıflara dönüştürelim
            emotion_results = []
            for res in results:
                label = res['label']
                score = res['score']
                
                # BERTweet etiketlerini duygu etiketlerine dönüştür
                if label == "POS":
                    if score > 0.8:
                        emotion = "joy"
                    else:
                        emotion = "optimism"
                elif label == "NEG":
                    if score > 0.8:
                        emotion = "anger"
                    else:
                        emotion = "sadness"
                else:  # NEU
                    emotion = "neutral"
                        
                emotion_results.append({"label": emotion, "score": score})
                
        except Exception as e:
            print(f"Error using transformers for sentiment analysis: {str(e)}")
            print("Falling back to TextBlob for sentiment analysis...")
            
            # Fallback to TextBlob
            emotion_results = []
            for sentence in sentences:
                analysis = TextBlob(sentence)
                
                # Map polarity to emotions (simplified)
                if analysis.polarity > 0.5:
                    emotion = "joy"
                elif analysis.polarity > 0.1:
                    emotion = "optimism"
                elif analysis.polarity < -0.5:
                    emotion = "anger"
                elif analysis.polarity < -0.1:
                    emotion = "sadness"
                else:
                    emotion = "neutral"
                
                emotion_results.append({"label": emotion, "score": abs(analysis.polarity)})
        
        # Save results to file
        with open(self.sentiment_results_path, "w", encoding="utf-8") as f_out:
            for sentence, res in zip(sentences, emotion_results):
                output = f"Cümle: {sentence}\nDuygu: {res['label']}, Güven: {res['score']:.2f}\n\n"
                f_out.write(output)
        
        # Create visualization
        labels = [res['label'] for res in emotion_results]
        label_counts = Counter(labels)
        
        plt.figure(figsize=(6, 6))
        plt.pie(
            label_counts.values(), 
            labels=label_counts.keys(), 
            autopct='%1.1f%%', 
            colors=['blue', 'red', 'orange', 'green', 'purple', 'pink']
        )
        plt.title("Toplantı Duygu Dağılımı")
        plt.savefig(self.sentiment_chart_path)
        
        print(f"Sentiment analysis results saved to {self.sentiment_results_path}")
        print(f"Sentiment chart saved to {self.sentiment_chart_path}")
        
        return emotion_results
    
    """
    def analyze_text_sentiment(self, text=None):
        #Analyze sentiment in text
        if text is None:
            try:
                with open(self.transcript_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                print(f"Error reading transcript: {str(e)}")
                return None
        
        print("Analyzing text sentiment...")
        
        # Split text into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        
        try:
            # Try using transformers pipeline
            from transformers import pipeline
            
            if self.sentiment_analyzer is None:
                print("Loading sentiment analysis model...")
                self.sentiment_analyzer = pipeline(
                    "text-classification", 
                    model="cardiffnlp/twitter-roberta-base-emotion",
                    device=-1  # Force CPU usage
                )
                
            results = [self.sentiment_analyzer(sentence)[0] for sentence in sentences]
        except Exception as e:
            print(f"Error using transformers for sentiment analysis: {str(e)}")
            print("Falling back to TextBlob for sentiment analysis...")
            
            # Fallback to TextBlob
            results = []
            for sentence in sentences:
                analysis = TextBlob(sentence)
                
                # Map polarity to emotions (simplified)
                if analysis.polarity > 0.5:
                    emotion = "joy"
                elif analysis.polarity > 0.1:
                    emotion = "optimism"
                elif analysis.polarity < -0.5:
                    emotion = "anger"
                elif analysis.polarity < -0.1:
                    emotion = "sadness"
                else:
                    emotion = "neutral"
                
                results.append({"label": emotion, "score": abs(analysis.polarity)})
        
        # Save results to file
        with open(self.sentiment_results_path, "w", encoding="utf-8") as f_out:
            for sentence, res in zip(sentences, results):
                output = f"Cümle: {sentence}\nDuygu: {res['label']}, Güven: {res['score']:.2f}\n\n"
                f_out.write(output)
        
        # Create visualization
        labels = [res['label'] for res in results]
        label_counts = Counter(labels)
        
        plt.figure(figsize=(6, 6))
        plt.pie(
            label_counts.values(), 
            labels=label_counts.keys(), 
            autopct='%1.1f%%', 
            colors=['blue', 'red', 'orange', 'green', 'purple', 'pink']
        )
        plt.title("Toplantı Duygu Dağılımı")
        plt.savefig(self.sentiment_chart_path)
        
        print(f"Sentiment analysis results saved to {self.sentiment_results_path}")
        print(f"Sentiment chart saved to {self.sentiment_chart_path}")
        
        return results
    """
    
    
    def analyze_audio_emotion(self, audio_path=None, visualize=True):
        """Analyze emotions in audio"""
        if audio_path is None:
            audio_path = self.audio_path
            
        print(f"Analyzing audio emotions from {audio_path}...")
        
        try:
            # Initialize the audio emotion analyzer if needed
            if self.audio_emotion_analyzer is None:
                self.audio_emotion_analyzer = AudioEmotionAnalyzer()
                
            # Analyze the audio
            result = self.audio_emotion_analyzer.predict_emotion(audio_path, visualize=visualize)
            
            if result:
                print(f"Predicted audio emotion: {result['emotion']} with {result['confidence']*100:.2f}% confidence")
                return result
            return None
        except Exception as e:
            print(f"Error analyzing audio emotions: {str(e)}")
            return None
    
    def run_full_pipeline(self, video_path, visualize=True):
        """Run the complete analysis pipeline on a video file"""
        print(f"Starting full analysis pipeline for: {video_path}")
        
        # Step 1: Extract audio
        if not self.extract_audio(video_path):
            print("Failed to extract audio. Pipeline stopped.")
            return False
        
        # Step 2: Transcribe audio to text
        transcript = self.transcribe_audio()
        if transcript is None:
            print("Failed to transcribe audio. Pipeline will continue with limited functionality.")
        
        # Step 3: Summarize text
        summary = self.summarize_text()
        
        # Step 4: Analyze text sentiment
        sentiment_results = self.analyze_text_sentiment()
        
       
        audio_emotion = self.analyze_audio_emotion(visualize=visualize)
        
        print("\nMedia Analysis Pipeline completed successfully!")
        return {
            "video_path": video_path,
            "audio_path": self.audio_path,
            "transcript_path": self.transcript_path,
            "summary_path": self.summary_path,
            "sentiment_results_path": self.sentiment_results_path,
            "sentiment_chart_path": self.sentiment_chart_path,
            "audio_emotion": audio_emotion
        }


import os
import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, pipeline

class AudioEmotionAnalyzer:
    """Class for analyzing emotions in audio files"""
    
    def __init__(self, model_name="superb/hubert-large-superb-er"):
        """
        Initialize the audio emotion analyzer with a pre-trained model from HuggingFace
        
        Args:
            model_name: Name or path of the pre-trained model
                Default is now "superb/hubert-large-superb-er" which is specialized for emotion recognition
                Other recommended models:
                - "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition" (original default)
                - "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim" (high accuracy)
                - "declare-lab/mbert-base-uncased-emotion" (text-based emotion analysis)
        """
        print(f"Loading audio emotion model: {model_name}")
        try:
            # Try loading the primary model
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(model_name)
            self.sampling_rate = self.feature_extractor.sampling_rate
            
            def change_emotion_labels(self, new_labels):
                """
                Change the emotion labels in the model's configuration
                
                Args:
                    new_labels: List of new emotion labels, should match the number of existing labels
                """
                if len(new_labels) != len(self.id2label):
                    raise ValueError(f"Number of new labels ({len(new_labels)}) must match existing labels ({len(self.id2label)})")
                
                # Update the id2label dictionary
                for i in range(len(new_labels)):
                    self.id2label[i] = new_labels[i]
                
                # Also update the label2id dictionary if it exists
                if hasattr(self.model.config, 'label2id'):
                    self.model.config.label2id = {label: idx for idx, label in self.id2label.items()}
                
                print(f"Updated emotion labels: {list(self.id2label.values())}")
            
            
            # Get emotion labels from model config
            self.id2label = self.model.config.id2label
            print(f"Available emotions: {list(self.id2label.values())}")
            
            # Create a pipeline for easy inference
            self.pipeline = pipeline(
                "audio-classification", 
                model=model_name,
                device=0 if torch.cuda.is_available() else -1  # Use GPU if available
            )
            print(f"Audio emotion model loaded successfully on {'GPU' if torch.cuda.is_available() else 'CPU'}")
            
        except Exception as e:
            print(f"Error loading primary audio emotion model: {str(e)}")
            print("Attempting to load backup model...")
            
            try:
                # Try loading the backup model - always use a proper emotion recognition model
                backup_model = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(backup_model)
                self.model = AutoModelForAudioClassification.from_pretrained(backup_model)
                self.sampling_rate = self.feature_extractor.sampling_rate
                self.id2label = self.model.config.id2label
                
                self.pipeline = pipeline(
                    "audio-classification", 
                    model=backup_model,
                    device=0 if torch.cuda.is_available() else -1
                )
                print(f"Backup audio emotion model loaded successfully")
                
            except Exception as e2:
                print(f"Error loading backup model: {str(e2)}")
                print("Will attempt to use a simplified approach if analysis is requested")
                self.model = None
                self.feature_extractor = None
                self.pipeline = None
                self.id2label = {
                    0: "angry", 1: "happy", 2: "neutral", 3: "sad",
                    4: "fearful", 5: "disgusted", 6: "surprised"
                }
                # For fallback mode, ensure we only use emotion labels, not command words
                self.sampling_rate = 16000  # Default sampling rate
    
    def preprocess_audio(self, file_path):
        """
        Preprocess audio file to match model requirements
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Processed audio array and original duration
        """
        try:
            # Check if file exists
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")
                
            # Handle different audio formats - convert to WAV if needed
            if not file_path.lower().endswith('.wav'):
                print(f"Converting {file_path} to WAV format...")
                converted_path = file_path.rsplit('.', 1)[0] + '.wav'
                sound = AudioSegment.from_file(file_path)
                sound.export(converted_path, format="wav")
                file_path = converted_path
            
            # Load and resample audio if needed
            audio, orig_sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=audio, sr=orig_sr)
            
            # Resample if needed
            if self.feature_extractor and orig_sr != self.sampling_rate:
                print(f"Resampling from {orig_sr} Hz to {self.sampling_rate} Hz")
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sampling_rate)
            
            return audio, duration, orig_sr
            
        except Exception as e:
            print(f"Error preprocessing audio file {file_path}: {str(e)}")
            return None, 0, 0
    
    def normalize_audio(self, audio):
        """
        Normalize audio to improve prediction accuracy
        
        Args:
            audio: Audio array
            
        Returns:
            Normalized audio array
        """
        # Apply normalization to ensure consistent volume levels
        if np.abs(audio).max() > 0:
            return audio / np.abs(audio).max() * 0.9
        return audio
    
    def predict_emotion(self, file_path, visualize=False, segment_length=None):
        """
        Predict emotion from audio file
        
        Args:
            file_path: Path to the audio file
            visualize: Whether to visualize the audio and prediction results
            segment_length: Optional length in seconds to segment audio for analysis
                            (useful for longer recordings)
            
        Returns:
            Dictionary with prediction results
            
        Note:
            This function always ensures proper emotion labels are used, even with fallback methods.
            Expected emotions are: angry, happy, sad, neutral, fearful, disgusted, surprised, etc.
            NOT command words like "stop", "go", "left", etc.
        """
        """
        Predict emotion from audio file
        
        Args:
            file_path: Path to the audio file
            visualize: Whether to visualize the audio and prediction results
            segment_length: Optional length in seconds to segment audio for analysis
                            (useful for longer recordings)
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess audio
            audio, duration, sample_rate = self.preprocess_audio(file_path)
            if audio is None:
                return None
                
            # Apply normalization
            audio = self.normalize_audio(audio)
            
            # If model and pipeline are available, use them for prediction
            if self.pipeline and self.model:
                if segment_length and duration > segment_length:
                    # For long audio files, segment and analyze parts separately
                    return self._analyze_segmented_audio(audio, sample_rate, file_path, duration, segment_length, visualize)
                else:
                    # For shorter files, analyze the whole file
                    return self._analyze_complete_audio(audio, sample_rate, file_path, duration, visualize)
            else:
                # Fallback: Use audio features for basic emotion classification
                print("Using fallback audio emotion detection...")
                features = self._extract_basic_features(audio, sample_rate)
                emotion, confidence = self._classify_with_features(features)
                
                result = {
                    'file': os.path.basename(file_path),
                    'emotion': emotion,
                    'confidence': confidence,
                    'duration': duration,
                    'all_emotions': {emotion: confidence}
                }
                
                # Visualize if requested
                if visualize:
                    self.visualize_prediction(audio, sample_rate, result)
                    
                return result
            
        except Exception as e:
            print(f"Error predicting emotion from {file_path}: {str(e)}")
            return None
    
    def _analyze_complete_audio(self, audio, sample_rate, file_path, duration, visualize):
        """Analyze a complete audio file without segmentation"""
                        # Use pipeline for direct prediction
        pipeline_result = self.pipeline({"raw": audio, "sampling_rate": sample_rate})
        
        # Check if results are emotions or word commands
        first_label = pipeline_result[0]['label'].lower()
        
        # If we got command words instead of emotions, use fallback method
        if first_label in ['stop', 'go', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off']:
            print("Warning: Model returned command words instead of emotions. Using fallback method.")
            features = self._extract_basic_features(audio, sample_rate)
            emotion, confidence = self._classify_with_features(features)
            
            # Create emotion scores dictionary
            all_emotions = {
                "angry": 0.1, "happy": 0.1, "sad": 0.1, "neutral": 0.1,
                "fearful": 0.1, "disgusted": 0.1, "surprised": 0.1
            }
            all_emotions[emotion] = confidence
            
            predicted_emotion = emotion
        else:
            # Get the highest scoring emotion
            predicted_emotion = first_label
            confidence = pipeline_result[0]['score']
            
            # Create a dictionary of all emotion scores
            all_emotions = {item['label']: item['score'] for item in pipeline_result}
        
        # Create detailed result
        result = {
            'file': os.path.basename(file_path),
            'emotion': predicted_emotion,
            'confidence': confidence,
            'duration': duration,
            'all_emotions': all_emotions
        }
        
        # Visualize if requested
        if visualize:
            self.visualize_prediction(audio, sample_rate, result)
            
        return result
    
    def _analyze_segmented_audio(self, audio, sample_rate, file_path, duration, segment_length, visualize):
        """Analyze a long audio file by segmenting it"""
        segment_samples = int(segment_length * sample_rate)
        num_segments = int(np.ceil(len(audio) / segment_samples))
        
        print(f"Analyzing {num_segments} segments of {segment_length}s each")
        
        segment_results = []
        
        for i in range(num_segments):
            # Extract segment
            start = i * segment_samples
            end = min(start + segment_samples, len(audio))
            segment = audio[start:end]
            
            # Skip very short segments
            if len(segment) < 0.5 * segment_samples:
                continue
                
            # Use pipeline for prediction
            pipeline_result = self.pipeline({"raw": segment, "sampling_rate": sample_rate})
            
            # Get segment result
            segment_result = {
                'start_time': start / sample_rate,
                'end_time': end / sample_rate,
                'emotion': pipeline_result[0]['label'],
                'confidence': pipeline_result[0]['score'],
                'all_emotions': {item['label']: item['score'] for item in pipeline_result}
            }
            
            segment_results.append(segment_result)
        
        # Aggregate results from all segments
        emotion_scores = {}
        for seg in segment_results:
            for emotion, score in seg['all_emotions'].items():
                if emotion not in emotion_scores:
                    emotion_scores[emotion] = []
                emotion_scores[emotion].append(score)
        
        # Calculate average scores
        avg_scores = {emotion: np.mean(scores) for emotion, scores in emotion_scores.items()}
        
        # Find the dominant emotion
        dominant_emotion = max(avg_scores.items(), key=lambda x: x[1])
        
        # Create final result
        result = {
            'file': os.path.basename(file_path),
            'emotion': dominant_emotion[0],
            'confidence': dominant_emotion[1],
            'duration': duration,
            'all_emotions': avg_scores,
            'segments': segment_results
        }
        
        # Visualize if requested
        if visualize:
            self.visualize_segmented_prediction(audio, sample_rate, result, segment_results)
            
        return result
    
    def _extract_basic_features(self, audio, sample_rate):
        """Extract basic audio features for fallback classification"""
        # Extract some basic audio features
        features = {}
        
        # Loudness/energy
        features['rms'] = np.mean(librosa.feature.rms(y=audio))
        
        # Spectral centroid (brightness)
        features['centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
        
        # Zero crossing rate (noisiness)
        features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        # Tempo estimation
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
        features['tempo'] = tempo
        
        # Additional features for better classification
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        
        # Chroma features (related to musical content)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        features['chroma_mean'] = np.mean(chroma, axis=1)
        
        return features
    
    def _classify_with_features(self, features):
        """Enhanced rule-based classification based on audio features"""
        # More nuanced emotion classification based on audio features
        # High energy, fast tempo -> happy/excited
        if features['rms'] > 0.1 and features['tempo'] > 120:
            return "happy", 0.7
        # High energy, high ZCR -> angry/intense
        elif features['rms'] > 0.1 and features['zcr'] > 0.1:
            return "angry", 0.65
        # Low energy, slow tempo -> sad/depressed
        elif features['rms'] < 0.05 and features['tempo'] < 100:
            return "sad", 0.7
        # Low energy, moderate ZCR -> fearful
        elif features['rms'] < 0.05 and 0.05 < features['zcr'] < 0.1:
            return "fearful", 0.6
        # High ZCR, high spectral centroid -> surprised
        elif features['zcr'] > 0.15 and features['centroid'] > 3000:
            return "surprised", 0.6
        # Low ZCR, low spectral centroid -> disgusted
        elif features['zcr'] < 0.05 and features['centroid'] < 1500:
            return "disgusted", 0.55
        # Default case
        else:
            return "neutral", 0.75
    
    def visualize_prediction(self, audio, sample_rate, result):
        """
        Visualize audio waveform and emotion prediction
        
        Args:
            audio: Audio array
            sample_rate: Audio sample rate
            result: Prediction result dictionary
        """
        plt.figure(figsize=(15, 10))
        
        # Plot waveform
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(audio, sr=sample_rate)
        plt.title(f"Audio Waveform - Predicted: {result['emotion']} ({result['confidence']*100:.2f}%)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        
        # Plot spectrogram
        plt.subplot(3, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Spectrogram")
        
        # Plot emotion probabilities
        plt.subplot(3, 1, 3)
        emotions = list(result['all_emotions'].keys())
        scores = list(result['all_emotions'].values())
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        emotions = [emotions[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        colors = ['#5DA5DA', '#FAA43A', '#60BD68', '#F17CB0', '#B2912F', '#B276B2', '#DECF3F', '#F15854']
        if len(colors) < len(emotions):
            colors = colors * (len(emotions) // len(colors) + 1)
        
        bars = plt.bar(range(len(emotions)), scores, color=colors[:len(emotions)])
        plt.xticks(range(len(emotions)), emotions, rotation=45, ha="right")
        plt.title("Emotion Probabilities")
        plt.xlabel("Emotion")
        plt.ylabel("Probability")
        plt.ylim(0, 1)
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height*100:.1f}%', ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        plt.savefig("audio_emotion_analysis.png")
        plt.show()
    
    def visualize_segmented_prediction(self, audio, sample_rate, result, segments):
        """
        Visualize segmented audio analysis
        
        Args:
            audio: Audio array
            sample_rate: Audio sample rate
            result: Overall prediction result dictionary
            segments: List of segment analysis results
        """
        plt.figure(figsize=(15, 12))
        
        # Plot waveform
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(audio, sr=sample_rate)
        plt.title(f"Audio Waveform - Overall Emotion: {result['emotion']} ({result['confidence']*100:.2f}%)")
        
        # Add segment boundaries and emotions
        for segment in segments:
            start = segment['start_time']
            end = segment['end_time']
            emotion = segment['emotion']
            conf = segment['confidence']
            
            plt.axvline(x=start, color='r', linestyle='--', alpha=0.5)
            plt.text(start + (end-start)/2, 0, 
                     f"{emotion}\n{conf*100:.1f}%",
                     ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.7))
        
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        
        # Plot spectrogram
        plt.subplot(3, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Spectrogram")
        
        # Plot emotion probabilities
        plt.subplot(3, 1, 3)
        emotions = list(result['all_emotions'].keys())
        scores = list(result['all_emotions'].values())
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        emotions = [emotions[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        colors = ['#5DA5DA', '#FAA43A', '#60BD68', '#F17CB0', '#B2912F', '#B276B2', '#DECF3F', '#F15854']
        if len(colors) < len(emotions):
            colors = colors * (len(emotions) // len(colors) + 1)
        
        bars = plt.bar(range(len(emotions)), scores, color=colors[:len(emotions)])
        plt.xticks(range(len(emotions)), emotions, rotation=45, ha="right")
        plt.title("Overall Emotion Probabilities")
        plt.xlabel("Emotion")
        plt.ylabel("Probability")
        plt.ylim(0, 1)
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height*100:.1f}%', ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        plt.savefig("audio_emotion_analysis_segmented.png")
        plt.show()

    def analyze_emotion_changes(self, file_path, segment_length=3.0, overlap=1.0):
        """
        Analyze emotion changes over time in an audio file
        
        Args:
            file_path: Path to the audio file
            segment_length: Length of each segment in seconds
            overlap: Overlap between segments in seconds
            
        Returns:
            Dictionary with emotion trajectory analysis
        """
        try:
            # Preprocess audio
            audio, duration, sample_rate = self.preprocess_audio(file_path)
            if audio is None:
                return None
                
            # Apply normalization
            audio = self.normalize_audio(audio)
            
            # Check if we have the necessary tools
            if not self.pipeline or not self.model:
                print("Advanced emotion trajectory analysis requires the model to be loaded")
                return None
                
            # Calculate segment size in samples
            segment_samples = int(segment_length * sample_rate)
            step_samples = int((segment_length - overlap) * sample_rate)
            
            # Split audio into overlapping segments
            segments = []
            segment_times = []
            
            for start in range(0, len(audio) - int(0.5 * segment_samples), step_samples):
                end = min(start + segment_samples, len(audio))
                segment = audio[start:end]
                segments.append(segment)
                segment_times.append(start / sample_rate)
            
            # Analyze each segment
            segment_results = []
            
            for i, (segment, start_time) in enumerate(zip(segments, segment_times)):
                # Calculate end time
                end_time = min(start_time + segment_length, duration)
                
                # Skip very short segments
                if len(segment) < 0.5 * segment_samples:
                    continue
                    
                # Use pipeline for prediction
                pipeline_result = self.pipeline({"raw": segment, "sampling_rate": sample_rate})
                
                # Get segment result
                segment_result = {
                    'segment_index': i,
                    'start_time': start_time,
                    'end_time': end_time,
                    'emotion': pipeline_result[0]['label'],
                    'confidence': pipeline_result[0]['score'],
                    'all_emotions': {item['label']: item['score'] for item in pipeline_result}
                }
                
                segment_results.append(segment_result)
            
            # Create emotion trajectories
            emotions = list(set([r['emotion'] for r in segment_results]))
            trajectories = {emotion: [] for emotion in emotions}
            
            for result in segment_results:
                for emotion in emotions:
                    score = result['all_emotions'].get(emotion, 0)
                    trajectories[emotion].append((result['start_time'], score))
            
            # Find emotional transitions
            transitions = []
            prev_emotion = segment_results[0]['emotion'] if segment_results else None
            
            for i, result in enumerate(segment_results[1:], 1):
                if result['emotion'] != prev_emotion:
                    transitions.append({
                        'time': result['start_time'],
                        'from': prev_emotion,
                        'to': result['emotion'],
                        'confidence': result['confidence']
                    })
                    prev_emotion = result['emotion']
            
            # Create final analysis result
            analysis = {
                'file': os.path.basename(file_path),
                'duration': duration,
                'segment_length': segment_length,
                'overlap': overlap,
                'num_segments': len(segment_results),
                'dominant_emotion': max([(emotion, len([r for r in segment_results if r['emotion'] == emotion])) 
                                       for emotion in emotions], key=lambda x: x[1])[0],
                'emotion_distribution': {emotion: len([r for r in segment_results if r['emotion'] == emotion]) / len(segment_results)
                                       for emotion in emotions},
                'transitions': transitions,
                'segments': segment_results,
                'trajectories': trajectories
            }
            
            # Visualize emotion trajectories
            self.visualize_emotion_trajectories(audio, sample_rate, analysis)
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing emotion changes in {file_path}: {str(e)}")
            return None
    
    def visualize_emotion_trajectories(self, audio, sample_rate, analysis):
        """
        Visualize emotion trajectories over time
        
        Args:
            audio: Audio array
            sample_rate: Audio sample rate
            analysis: Emotion trajectory analysis result
        """
        plt.figure(figsize=(15, 12))
        
        # Plot waveform
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(audio, sr=sample_rate)
        plt.title(f"Audio Waveform - Dominant Emotion: {analysis['dominant_emotion']}")
        
        # Add transition markers
        for transition in analysis['transitions']:
            plt.axvline(x=transition['time'], color='r', linestyle='--')
            plt.text(transition['time'], 0, f"{transition['from']} → {transition['to']}",
                     rotation=90, va='center', ha='right',
                     bbox=dict(facecolor='white', alpha=0.7))
        
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        
        # Plot emotion trajectories
        plt.subplot(3, 1, 2)
        colors = ['#5DA5DA', '#FAA43A', '#60BD68', '#F17CB0', '#B2912F', '#B276B2', '#DECF3F', '#F15854']
        
        for i, (emotion, trajectory) in enumerate(analysis['trajectories'].items()):
            if trajectory:  # Check if trajectory is not empty
                times, scores = zip(*trajectory)
                plt.plot(times, scores, label=emotion, color=colors[i % len(colors)],
                         marker='o', markersize=4, linewidth=2)
        
        plt.xlabel("Time (s)")
        plt.ylabel("Emotion Score")
        plt.title("Emotion Trajectories Over Time")
        plt.legend()
        plt.grid(True)
        
        # Plot emotion distribution
        plt.subplot(3, 1, 3)
        emotions = list(analysis['emotion_distribution'].keys())
        distribution = list(analysis['emotion_distribution'].values())
        
        # Sort by distribution value
        sorted_indices = np.argsort(distribution)[::-1]
        emotions = [emotions[i] for i in sorted_indices]
        distribution = [distribution[i] for i in sorted_indices]
        
        bars = plt.bar(range(len(emotions)), distribution, 
                      color=[colors[i % len(colors)] for i in range(len(emotions))])
        plt.xticks(range(len(emotions)), emotions, rotation=45, ha="right")
        plt.title("Emotion Distribution")
        plt.xlabel("Emotion")
        plt.ylabel("Proportion of Time")
        plt.ylim(0, 1)
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height*100:.1f}%', ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        plt.savefig("audio_emotion_trajectories.png")
        plt.show()


def fix_numpy_dependencies():
    """
    Fix NumPy dependency issues by ensuring compatible versions
    """
    try:
        import numpy as np
        numpy_version = np.__version__
        print(f"Current NumPy version: {numpy_version}")
        
        # Check if NumPy 2.x is installed
        if numpy_version.startswith('2.'):
            print("NumPy 2.x detected, which may cause compatibility issues.")
            print("Attempting to fix by downgrading NumPy...")
            
            # Try to downgrade NumPy
            import subprocess
            subprocess.check_call(["pip", "install", "numpy==1.26.4", "--force-reinstall"])
            print("NumPy downgraded to 1.26.4. You may need to restart your script.")
            print("After restarting, run pip install scipy scikit-learn transformers --force-reinstall")
            
            return False
        
        return True
    except Exception as e:
        print(f"Error checking NumPy version: {str(e)}")
        return False


def main():
    """Main function to run the complete pipeline"""
    # Check and fix NumPy dependencies
    if not fix_numpy_dependencies():
        print("Please restart the script after fixing NumPy dependencies.")
        return
    
    # Create the pipeline
    pipeline = MediaAnalysisPipeline()
    
    # Get video path from user
    video_path = input("Enter the path to your video file: ")
    
    # Run the full pipeline
    result = pipeline.run_full_pipeline(video_path)
    
    if result:
        print("\nAnalysis Summary:")
        print(f"Video: {result['video_path']}")
        print(f"Audio: {result['audio_path']}")
        print(f"Transcript: {result['transcript_path']}")
        print(f"Summary: {result['summary_path']}")
        print(f"Sentiment Analysis: {result['sentiment_results_path']}")
        print(f"Sentiment Chart: {result['sentiment_chart_path']}")
        
        if result['audio_emotion']:
            print(f"Audio Emotion: {result['audio_emotion']['emotion']} "
                  f"({result['audio_emotion']['confidence']*100:.2f}%)")


if __name__ == "__main__":
    main()