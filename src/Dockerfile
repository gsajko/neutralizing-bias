FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
RUN apt-get update -y
RUN apt-get install -y python 
RUN apt-get install -y python-pip build-essential
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY . /app
CMD ["python", "download_punkt.py"]
CMD ["sh" "download_data_ckpt_and_run_inference.sh"]
CMD ["python", "api/app.py"]
