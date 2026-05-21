FROM skhatiri/aerialist:latest
RUN apt-get update && apt-get install -y --no-install-recommends xvfb && rm -rf /var/lib/apt/lists/*
RUN pip3 install -e /src/aerialist/ lz4

RUN wget https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0-x86_64.tar.gz && \
    tar -xzf patchelf-0.18.0-x86_64.tar.gz && \
    mv bin/patchelf /usr/local/bin/ && \
    rm -rf patchelf-0.18.0-x86_64.tar.gz bin share

COPY ./requirements.txt /src/generator/requirements.txt
WORKDIR /src/generator/
RUN pip3 install -r /src/generator/requirements.txt


ENV AGENT=local
ENV AVOIDANCE_LAUNCH=/src/aerialist/aerialist/resources/simulation/collision_prevention.launch
ENV AVOIDANCE_BOX=/src/aerialist/aerialist/resources/simulation/box.xacro
ENV AVOIDANCE_WORLD=collision_prevention
ENV AVOIDANCE_WORLD_PATH=/src/aerialist/aerialist/resources/simulation/collision_prevention.world
