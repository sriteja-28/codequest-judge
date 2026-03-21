FROM gcc:13-bookworm

RUN useradd -r -s /bin/false -u 65534 nobody 2>/dev/null || true

WORKDIR /code

# g++ is already available in the base image

CMD ["/bin/sh", "-c", "g++ -O2 -std=c++17 -o /tmp/sol /code/solution.cpp && /tmp/sol"]