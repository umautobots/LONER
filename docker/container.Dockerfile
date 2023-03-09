ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:21.10-py3

FROM ${BASE_IMAGE}

ARG USER_NAME=zed
ARG USER_ID=1000


# Prevent anything requiring user input
ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=linux

ENV TZ=America
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# # Hack to update just enough to install wget
# RUN (apt-get -y update || true) && apt-get -y install wget
# # Fix keyring
# RUN apt-key del 7fa2af80 \
#   && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
#   && dpkg -i cuda-keyring_1.0-1_all.deb \
#   && rm /etc/apt/sources.list.d/cuda.list \
#   && wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - \
#   && apt-get -y update

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
     && pip3 install opencv-python pykitti \
     && pip3 install --user git+https://github.com/DanielPollithy/pypcd.git \
     && pip3 install rospkg \
     && pip3 install pycryptodomex \
     && pip3 install gnupg \
     && pip3 install opencv-python==4.5.5.64 \
     && pip3 install open3d \
     && pip3 install autopep8 \
     && pip3 install torch_tb_profiler \
     && pip3 install torchviz \
     && pip3 install --upgrade typing-extensions \
     && rm /tmp/requirements.txt

# Install tiny-cuda-nn
RUN ldconfig && pip3 install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch || \
    (echo "Note: Unable find Cuda. See the README Build Section for details on fixing" && false)

RUN ldconfig && pip3 install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.2"

RUN useradd -m -l -u ${USER_ID} -s /bin/bash ${USER_NAME} \
    && usermod -aG video ${USER_NAME}

# Give them passwordless sudo
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Switch to user to run user-space commands
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

RUN sudo rosdep init && rosdep update

# finish ROS setup
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# This overrides the default CarlaSim entrypoint, which we want. Theirs starts the simulator.
COPY ./entrypoint.sh /entrypoint.sh
RUN sudo chmod +x /entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]