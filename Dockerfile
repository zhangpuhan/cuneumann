FROM nvidia/cuda:11.2.0-devel-ubuntu20.04 AS builds


# installing requirements to get and extract prebuilt binaries
RUN apt-get update && apt-get install -y \
 xz-utils \
 curl \
 && rm -rf /var/lib/apt/lists/*

# Getting prebuilt binary from llvm 
RUN curl -SL https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.0/clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz \
 | tar -xJC . && \
 mv clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-20.04 clang_11

RUN apt-get update && apt-get install -y wget\
 && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/Kitware/CMake/releases/download/v3.19.3/cmake-3.19.3-Linux-x86_64.tar.gz && \ 
 tar -xvzf cmake-3.19.3-Linux-x86_64.tar.gz && \
 mv cmake-3.19.3-Linux-x86_64 cmake && \
 rm -rf cmake-3.19.3-Linux-x86_64.tar.gz

# Setting env
ENV PATH /clang_11/bin:$PATH
ENV LD_LIBRARY_PATH /clang_11/lib:$LD_LIBRARY_PATH
ENV PATH /cmake/bin:$PATH

# Compile
WORKDIR /opt/cuneumann
COPY CMakeLists.txt main.cpp model.cuh model.cu util.h kernel.cuh kernel.cu ./
RUN mkdir build && cd build && cmake .. && make


FROM ubuntu:20.04
COPY --from=builds /opt/cuneumann/build/cuneumann /opt/cuneumann/cuneumann

COPY --from=builds \
    /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.11.0 \
    /usr/local/cuda/targets/x86_64-linux/lib/libcusparse.so.11 \
    /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.11 \
    /usr/local/cuda/targets/x86_64-linux/lib/libcublasLt.so.11 \
    /usr/local/cuda/targets/x86_64-linux/lib/

COPY --from=builds /lib64/ld-linux-x86-64.so.2 /lib64/ld-linux-x86-64.so.2

COPY --from=builds \
    /lib/x86_64-linux-gnu/libstdc++.so.6 \
    /lib/x86_64-linux-gnu/libm.so.6 \
    /lib/x86_64-linux-gnu/libgcc_s.so.1 \
    /lib/x86_64-linux-gnu/libc.so.6 \
    /lib/x86_64-linux-gnu/libdl.so.2 \
    /lib/x86_64-linux-gnu/libpthread.so.0 \
    /lib/x86_64-linux-gnu/librt.so.1 \
    /lib/x86_64-linux-gnu/

ENV PATH /opt/cuneumann:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH

LABEL maintainer=pz4ee@virginia.edu
ENTRYPOINT ["cuneumann"]