cd


export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6/pyrealsense2
source ~/.bashrc


echo '
______________________________________________
|                                            |
|              .,-:;//;:=,                   |
|            . :H@@@MM@M#H/.,+%;,            |
|         ,/X+ +M@@M@MM%=,-%HMMM@X/,         |
|       -+@MM; $M@@MH+-,;XMMMM@MMMM@+-       |
|      ;@M@@M- XM@X;. -+XXXXXHHH@M@M#@/.     |
|    ,%MM@@MH ,@%=            .---=-=:=,.    |
|    =@#@@@MX .,              -%HX$$%%%+;    |
|   =-./@M@M$                  .;@MMMM@MM:   |
|   X@/ -$MM/                    .+MM@@@M$   |
|  ,@M@H: :@:                    . =X#@@@@-  |
|  ,@@@MMX, .                    /H- ;@M@M=  |
|  .H@@@@M@+,                    %MM+..%#$.  |
|   /MMMM@MMH/.                  XM@MH; =;   |
|    /%+%$XHH@$=              , .H@@@@MX,    |
|     .=--------.           -%H.,@@@@@MX,    |
|     .%MM@@@HHHXX$$$%+- .:$MMX =M@@MM%.     |
|       =XMMM@MM@MM#H;,-+HMM@M+ /MMMX=       |
|         =%@M@M#@$-.=$@MM@@@M; %M%=         |
|           ,:+$+-,/H#MMMMMMM@= =,           |
|                 =++%%%%+/:-.               |
|____________________________________________|

'


cd

cd MAPPER_AGRI_MULTICAM/

git pull

echo  "AGRI MAPPER UPDATED SUCCESFULLY"
echo "STARTING VISION SYSTEM "



echo ""

echo "       _                 _         "
echo " _   _| |__  _   _ _ __ | |_ _   _ "
echo "| | | | '_ \| | | | '_ \| __| | | |"
echo "| |_| | |_) | |_| | | | | |_| |_| |"
echo " \__,_|_.__/ \__,_|_| |_|\__|\__,_|"

echo ""



python3 advanced_main.py
