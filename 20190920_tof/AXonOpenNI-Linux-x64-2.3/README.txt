# There are two zip files, one is for 32bit machine, the other one is for 64bit

# We choose 64bit(x64) and make the example as follows:

# The AXon Driver need libudev and libusb libaries, please install:

$ sudo apt-get install libudev-dev libusb-1.0-0-dev

# To run visual samples(e.g., SimpleViewer), you will need freeglut3 header and libaries, please install:

$ sudo apt-get install build-essential freeglut3 freeglut3-dev

# copy tgz file to any place you want(e.g., Home)

# unzip tgz file
$ tar zxvf OpenNI-Linux-x64-2.2.tar.bz2
$ cd OpenNI-Linux-x64-2.2

# run install.sh to generate OpenNIDevEnvironment, which contains OpenNI development environment 
#(sudo chmod a+x install.sh)

$ sudo ./install.sh

# please replug in the device for usb-register

# add environment variables
$ source OpenNIDevEnvironment

# build sample(e.g., AXon)
$ cd Samples/AXon
$ make

# run sample
# connect sensor
$ cd Bin/x64-Release
$ ./AXon

# now you should be able to see a GUI window showing the depth stream video

# If the Debian Jessie Lite is used for testing, it may require the following installation for properly start the viewer.

$ sudo apt-get install libgl1-mesa-dri

Cross-Compiling for ARM on Linux:
  The following environment variables should be defined:
  - OPENNI2_INCLUDE=<path SDK include dir>
  - OPENNI2_REDIST=<path SDK redist dir>
  - ARM_CXX=<path to cross-compilation g++>
  - ARM_STAGING=<path to cross-compilation staging dir>
  Then, run:
  $ PLATFORM=Arm make

