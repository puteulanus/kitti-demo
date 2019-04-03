FROM quay.io/puteulanus/nvcaffe-cpu as mozjpeg

RUN apt-get update && apt-get -y install build-essential nasm libtool pkg-config autoconf wget
RUN cd /tmp && \
    wget https://github.com/mozilla/mozjpeg/archive/v3.3.1.tar.gz && \
    tar zxf v3.3.1.tar.gz && \
    cd mozjpeg-3.3.1 && \
    autoreconf -i && \
    ./configure && \
    make -j"$(nproc)" && \
    make install

FROM quay.io/puteulanus/nvcaffe-cpu

COPY --from=mozjpeg /opt/mozjpeg /usr/local

RUN echo /usr/local/lib64 > /etc/ld.so.conf.d/libc64.conf && \
    ldconfig

RUN pip install gunicorn

ADD caffe /caffe
ADD car.py /caffe/car.py
ADD run.py /caffe/run.py

WORKDIR /caffe

EXPOSE 8080

CMD gunicorn -b 0.0.0.0:8080 run:app