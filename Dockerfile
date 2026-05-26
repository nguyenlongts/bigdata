FROM eclipse-temurin:17-jdk-jammy

RUN apt-get update && \
    apt-get install -y python3 python3-pip procps && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

ENV JAVA_HOME=/opt/java/openjdk
ENV PYSPARK_PYTHON=python3

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["tail", "-f", "/dev/null"]