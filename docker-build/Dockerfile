FROM python:3.9-slim

WORKDIR /opt
RUN apt-get update && apt-get install -y git
RUN pip install --upgrade pip
RUN pip install numpy \
				pandas \
				scipy \
				matplotlib \
				seaborn \
				scikit-learn \
				torch \
				torchvision \
				torch_summary \
				pytorch-pfn-extras


WORKDIR /work

###
# docker build -t sho/pytorch:1.10.2 .
# docker run -it --name vae_mnist -v ~/git-portfolio/Dockerfile/pytorch/:/work <Image ID> bash
