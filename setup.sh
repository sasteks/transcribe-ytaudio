# 1. Update package lists and install FFmpeg and Git
sudo apt-get update && sudo apt-get install -y ffmpeg git python3-venv python3-pip

# 2. Create and activate a clean Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install PyTorch with CUDA support (for NVIDIA GPUs)
# Note: If running CPU-only, remove the --index-url flag
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install Whisper, Transformers, Accelerate, and document export packages
pip install git+https://github.com/openai/whisper.git yt-dlp transformers accelerate python-docx