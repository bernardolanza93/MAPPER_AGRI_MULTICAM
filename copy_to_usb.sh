cd

export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6/pyrealsense2
source ~/.bashrc


cd

cd MAPPER_AGRI_MULTICAM/

cd aquisition
echo ""
echo -n "MEDIA IN ACQUISITION FOLDER:"
pwd
echo ""
ls
echo ""

cd ..

cd data/

echo -n "FILE IN DATA FOLDER: "
pwd
echo ""

ls
echo ""

cd

cd
cd MAPPER_AGRI_MULTICAM/
git pull

echo "COPY DATA TO USB"


echo "MMT - LAB"
echo ""

echo "       _                 _         "
echo " _   _| |__  _   _ _ __ | |_ _   _ "
echo "| | | | '_ \| | | | '_ \| __| | | |"
echo "| |_| | |_) | |_| | | | | |_| |_| |"
echo " \__,_|_.__/ \__,_|_| |_|\__|\__,_|"

echo ""

echo "RUNNING:  upload_on_usb.py"

python3 upload_on_usb.py
