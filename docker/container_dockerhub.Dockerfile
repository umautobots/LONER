ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:21.10-py3

FROM ${BASE_IMAGE}

# Prevent anything requiring user input
ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=linux

ENV TZ=America
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Basic packages
RUN apt-get -y update \
    && apt-get -y install \
      python3-pip \
      sudo \
      vim \
      wget \
      curl \
      software-properties-common \
      doxygen \
    && rm -rf /var/lib/apt/lists/*

# Install ROS noetic (desktop full)
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

RUN apt-get -y update \
    && apt-get -y install ros-noetic-desktop-full \
    && rm -rf /var/lib/apt/lists/*  

# Auxillary ROS installs
RUN apt-get -y update \
    && apt-get -y install \ 
      python3-rosdep \
      python3-rosinstall \
      python3-rosinstall-generator \
      python3-wstool \
      build-essential \
      python3-catkin-tools \
      ros-noetic-ros-numpy \
      ros-noetic-derived-object-msgs \
      ros-noetic-ackermann-msgs \
      ros-noetic-hector-trajectory-server \
    && rm -rf /var/lib/apt/lists/*  

# Extra misc installs
RUN apt-get -y update \
    && sudo apt-get -y install \ 
      libomp-dev \
      mesa-utils \
      apt-utils \
    && rm -rf /var/lib/apt/lists/*  

# Cloner-specific installs
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt \
     && pip3 install --user git+https://github.com/DanielPollithy/pypcd.git \
     && pip3 install --upgrade typing-extensions \
     && pip install evo --upgrade --no-binary evo \
     && rm /tmp/requirements.txt

# Install tiny-cuda-nn
RUN ldconfig && pip3 install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch || \
   (echo "Note: Unable find Cuda. See the README Build Section for details on fixing" && false)

RUN ldconfig && pip3 install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.2"

