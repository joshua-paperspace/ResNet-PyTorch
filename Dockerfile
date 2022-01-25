FROM nvcr.io/nvidia/pytorch:21.10-py3

# RUN apt-get update -y && \
#     apt-get install -y python-pip python-dev

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install -U pip
RUN pip3 install -r requirements.txt

EXPOSE 8501

COPY . /app

CMD ["streamlit", "run", "app2.py"]

# "--host=0.0.0.0"
# ENTRYPOINT [ "python3" ]

# CMD [ "app.py" ]