FROM python:3.7.13

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./ /app

# CMD ["python", "stan_compile.py"]

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

# Create a shell script to run multiple commands
RUN echo "#!/bin/sh\npython stan_compile.py\nuvicorn main:app --host 0.0.0.0 --port 80" > start.sh

# Give execution permission to the script
RUN chmod +x start.sh

# Use the shell script as CMD
CMD ["./start.sh"]