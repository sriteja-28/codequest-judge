FROM eclipse-temurin:21-jdk-jammy

RUN useradd -r -s /bin/false -u 65534 nobody 2>/dev/null || true

WORKDIR /code

CMD ["/bin/sh", "-c", "javac /code/Solution.java -d /tmp && java -cp /tmp Solution"]