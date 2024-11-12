@REM 1. install Repo
git clone https://github.com/LksWllmnn/visual_navigation_3dgs.git

@REM 2. create Virtual Environment
python -m venv .venv

@REM 2. install segemnt anything
pip install git+https://github.com/facebookresearch/segment-anything.git

@REM 3. install needed librarys
.venv/Scripts/activate
pip install opencv-python matplotlib 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118