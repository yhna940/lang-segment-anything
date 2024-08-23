FROM pytorch/pytorch:latest

# prevent apt from hanging
ARG DEBIAN_FRONTEND=noninteractive

# create workspace
ENV HOME /workspace
WORKDIR $HOME

# dependency: lang-segment-anything
RUN apt update

# installing system dependencies:
RUN apt install -y git
RUN apt install libgl1-mesa-glx -y
RUN apt install libglib2.0-0 -y

# copy source code:
COPY . $HOME/lang-segment-anything

# installing python dependencies:
WORKDIR $HOME/lang-segment-anything
RUN pip install -e .
