FROM armaneshaghi/synthseg:latest
#RUN git clone https://github.com/neuropoly/totalsegmentator-mri.git
COPY . /opt/totalsegmentator-mri
WORKDIR /opt/

