# AI Face-Capture for First-time Passport Application
Abstract: This paper presents the development and implementation of a Pan-Tilt-Zoom (PTZ) camera interfaced with a Jetson processor hosting Artificial Intelligence (AI) algorithms, designed to address the challenges of traditional biometric capture for first-time passport photos of young children. The system enhances the accuracy, efficiency, and adaptability of facial biometric capture by recognizing and accommodating the dynamic and diverse behaviors of children. Utilizing AI, the system detects faces and analyzes facial features using 68 landmark points, ensuring high-quality images that meet ISO standards. These technical specifications, implementation process, and performance evaluations of the developed system are provided in this paper, demonstrating the potential of the proposed AI Face-Capture to advance biometric identification and verification for young children.

# Overview
<br />
Face quality checking thresholds are in the thresholds.py file
Error messages are in the messages.py file
<br />
<b>Note:</b> Thresholds are established based on our experiment setup and may vary across different setups. Adjust the thresholds according to your specific requirements.

# Install dlib for python 3.6 on Jetson Nano
```console
wget http://dlib.net/files/dlib-19.21.tar.bz2
tar jxvf dlib-19.17.tar.bz2
cd dlib-19.21/
mkdir build
cd build/
#cmake ..
cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
cmake --build .
cd ../
sudo python3 setup.py install
```

# To get the environment ready
Install required packages
```console
git clone https://github.com/baset-sarker/ptz-first-time-passport.git
cd ptz-first-time-passport
pip install -r requirements.txt
```

# To run the code 
```console
python3 face_image_capture.py -i 1 # here 1 is for smbus port for jetson nano 
```

# Other repositories to check 
<b> Arducam MIPI </b>: https://github.com/ArduCAM/MIPI_Camera.git

<b> An Open-Source Face-Aware Capture System </b> : https://github.com/baset-sarker/face-aware-capture



# To cite the paper
```console
@Article{electronics13071178,
    AUTHOR = {Sarker, Md Abdul Baset and Hossain, S. M. Safayet and Venkataswamy, Naveenkumar G. and Schuckers, Stephanie and Imtiaz, Masudul H.},
    TITLE = {An Open-Source Face-Aware Capture System},
    JOURNAL = {Electronics},
    VOLUME = {13},
    YEAR = {2024},
    NUMBER = {7},
    ARTICLE-NUMBER = {1178},
    URL = {https://www.mdpi.com/2079-9292/13/7/1178},
    ISSN = {2079-9292},
    DOI = {10.3390/electronics13071178}
}


@article{202406.0392,
	doi = {10.20944/preprints202406.0392.v1},
	url = {https://doi.org/10.20944/preprints202406.0392.v1},
	year = 2024,
	month = {June},
	publisher = {Preprints},
	author = {MD ABDUL BASET SARKER and Masudul H. Imtiaz},
	title = {AI Face-Capture for First-time Passport Application},
	journal = {Preprints}
}

```
