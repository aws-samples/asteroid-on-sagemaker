FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.5.0-cpu-py36-ubuntu16.04

RUN apt-get update 
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*
    
RUN wget http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.28.tar.gz
RUN tar -xzf libsndfile-1.0.28.tar.gz
RUN cd libsndfile-1.0.28 && ./configure --prefix=/usr --disable-static --docdir=/usr/share/doc/libsndfile-1.0.28 && make && make install
RUN pip install pysoundfile numpy numba Cython asteroid sagemaker-inference PyYAML pytorch-lightning==1.4.9 flask gunicorn

ENV PATH="/opt/ml/code:${PATH}"

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

COPY ./ /opt/ml/code
WORKDIR /opt/ml/code
