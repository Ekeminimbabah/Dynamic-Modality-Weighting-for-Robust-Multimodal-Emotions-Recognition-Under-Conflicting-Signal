# 🚀 Quick Start Guide - Emotion Recognition Streamlit App

## ⚡ 30-Second Setup

### Option 1: Windows Users (Easiest)
1. Double-click: `run_streamlit.bat`
2. Wait for "http://localhost:8501" message
3. Open your browser to that URL
4. Done! 🎉

### Option 2: Mac/Linux Users
```bash
bash run_streamlit.sh
```

### Option 3: Manual Setup (Any OS)
```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements_streamlit.txt

# 4. Run the app
streamlit run app.py
```

---

## 📋 Pre-Requisites

Before running, make sure you have:

### ✅ Required Files
- [ ] Python 3.8 or higher installed
- [ ] Trained models in `checkpoints/` folder:
  - [ ] `face_iemocap.pth`
  - [ ] `speech_iemocap.pth`
  - [ ] `fusion_model.pth`

### ✅ System Requirements
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- GPU optional but recommended

### ✅ Check Your Setup
```bash
# Verify Python version
python --version  # Should be 3.8+

# Check if pip works
pip --version

# Verify checkpoints exist
ls checkpoints/  # or dir checkpoints/ on Windows
```

---

## 🎯 Deployment Options

### Option A: Local Development (Default)
```bash
streamlit run app.py
```
- Access at: `http://localhost:8501`
- Best for: Testing, development, personal use

### Option B: Local Network Access
```bash
streamlit run app.py --server.address=0.0.0.0
```
- Access from other computers: `http://your-ip:8501`
- Best for: Hospital network deployment

### Option C: Docker Container
```bash
# Build and run with Docker
docker build -t emotion-recognition:latest .
docker run -p 8501:8501 -v $(pwd)/checkpoints:/app/checkpoints emotion-recognition:latest
```

### Option D: Docker Compose (Recommended for Production)
```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```
- Access at: `http://localhost:8501`
- Persistent volumes for checkpoints and data
- Auto-restart on failure

---

## 📱 Using the Application

### Step 1: Open the App
Navigate to `http://localhost:8501` in your browser

### Step 2: Choose Input Type
- 📸 **Upload Image**: JPG or PNG with a clear face
- 🎤 **Upload Audio**: WAV or MP3 audio file
- 🎥 **Upload Video**: MP4 or MOV file (coming soon)

### Step 3: Upload Your File
Click "Browse files" and select your input

### Step 4: Analyze
Click the "🔍 Analyze Emotion" button

### Step 5: View Results
- See confidence scores for each emotion
- View dominant emotion with emoji
- Interactive charts for visualization

---

## 🎨 Customizing the Interface

### Change Colors
Edit `app.py` line ~40:
```python
EMOTION_COLORS = {
    0: '#f5576c',   # Angry - change to desired color
    1: '#ff9d56',   # Happy
    2: '#56ab91',   # Neutral
    3: '#764ba2'    # Sad
}
```

### Change Title
Edit `app.py` line ~20:
```python
st.set_page_config(
    page_title="Your Custom Title",  # Change this
    page_icon="🧠"
)
```

### Add Hospital Logo
Add after the header section in `app.py`:
```python
st.image("path/to/your/logo.png", width=200)
```

---

## 🔧 Troubleshooting

### Problem: "streamlit command not found"
```bash
# Solution:
pip install streamlit
# Or activate your virtual environment first
```

### Problem: "Models not found"
```bash
# Check that you're in the right directory
# and files exist:
ls checkpoints/
# Should show: face_iemocap.pth, speech_iemocap.pth, fusion_model.pth
```

### Problem: "Port 8501 already in use"
```bash
# Use a different port:
streamlit run app.py --server.port=8502
```

### Problem: "CUDA out of memory"
```bash
# The app will automatically fallback to CPU
# Or reduce batch size if needed
```

### Problem: Slow performance
- Ensure you're using GPU (CUDA)
- Check system resources
- Reduce input file sizes
- Use SSD instead of HDD for faster I/O

---

## 📊 Expected Performance

| Hardware | Image Analysis | Audio Analysis | Speed |
|----------|:-------------:|:-------------:|------:|
| CPU (i5) | 2-3s | 1-2s | Baseline |
| CPU (i7) | 1-2s | 0.5-1s | 2x faster |
| GPU (RTX 2060) | 0.5s | 0.3s | 5-10x faster |
| GPU (RTX 3060) | 0.3s | 0.2s | 10-15x faster |

---

## 📚 For Hospital IT Staff

### Network Deployment
```bash
# Run on hospital network
streamlit run app.py --server.address=192.168.x.x --server.port=8501
```

### Setting up Reverse Proxy (Nginx)
```nginx
server {
    listen 80;
    server_name emotion-recognition.hospital.local;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### With HTTPS/SSL
```nginx
server {
    listen 443 ssl;
    server_name emotion-recognition.hospital.local;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8501;
    }
}
```

---

## 🔒 Security Considerations

### Local Use (Development)
No special configuration needed

### Hospital Network Use
```bash
# Ensure secure connection
# Add authentication if needed
# Use firewall to restrict access
# Keep models in secure location
```

### Data Privacy
- ✅ All processing happens locally
- ✅ No data sent to external servers
- ✅ Files deleted after processing
- ✅ No data retention by default

---

## 🆘 Getting Help

1. **Check the full documentation**: See `STREAMLIT_README.md`
2. **Review logs**: Streamlit shows detailed error messages
3. **Test with sample files**: Use simple images/audio first
4. **Verify installations**: Run diagnostic commands above
5. **Check requirements**: Ensure all dependencies are installed

---

## 📞 Support Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **PyTorch Docs**: https://pytorch.org
- **Docker Docs**: https://docs.docker.com
- **OpenCV Docs**: https://docs.opencv.org

---

## ✨ Next Steps

1. ✅ Start the application
2. 📸 Test with sample image
3. 🎤 Test with sample audio
4. 📊 Verify predictions are reasonable
5. 🚀 Deploy to your environment

**Good luck! 🎉**

---

**Last Updated**: 2024
**Version**: 1.0.0
