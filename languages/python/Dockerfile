FROM python:3.12-slim

# Create non-root user for security
RUN useradd -r -s /bin/false -u 65534 nobody 2>/dev/null || true

# No pip packages needed for pure Python execution
# Add commonly used stdlib-only packages for competitive programming

WORKDIR /code

# Drop all capabilities, run as nobody
USER nobody

CMD ["python3", "-u", "/code/solution.py"]