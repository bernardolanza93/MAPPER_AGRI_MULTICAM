cd

export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6/pyrealsense2
source ~/.bashrc

cd
cd MAPPER_AGRI_MULTICAM/
git pull

echo "TEST ALL CAM "


echo "MMT - LAB"
echo ""

echo "       _                 _         "
echo " _   _| |__  _   _ _ __ | |_ _   _ "
echo "| | | | '_ \| | | | '_ \| __| | | |"
echo "| |_| | |_) | |_| | | | | |_| |_| |"
echo " \__,_|_.__/ \__,_|_| |_|\__|\__,_|"

echo ""

echo "RUNNING:  embbeded_simple_camera_display.py"

python3 embbeded_simple_camera_display.py
