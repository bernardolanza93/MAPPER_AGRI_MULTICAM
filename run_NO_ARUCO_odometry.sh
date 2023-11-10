cd

export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6/pyrealsense2
source ~/.bashrc

cd
cd MAPPER_AGRI_MULTICAM/
git pull


echo "NO ARUCO MODULE || ONLY T265"
echo  "AGRI MAPPER UPDATED SUCCESFULLY"
echo "STARTING STANDALONE ODOMETRY SYSTEM "


echo "MMT - LAB"
echo ""

echo "       _                 _         "
echo " _   _| |__  _   _ _ __ | |_ _   _ "
echo "| | | | '_ \| | | | '_ \| __| | | |"
echo "| |_| | |_) | |_| | | | | |_| |_| |"
echo " \__,_|_.__/ \__,_|_| |_|\__|\__,_|"

echo ""

echo "RUNNING:  embedded_odometry_NO_ARUCO.py"


python3 embedded_odometry_NO_ARUCO.py
