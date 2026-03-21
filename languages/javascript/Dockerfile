FROM node:20-slim

RUN useradd -r -s /bin/false -u 65534 nobody 2>/dev/null || true

WORKDIR /code

CMD ["node", "/code/solution.js"]