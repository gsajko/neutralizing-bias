FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
RUN apt-get update -y
RUN apt-get install -y python-dev 
RUN apt-get install -y python-pip build-essential locales
RUN apt-get install wget unzip
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY . /app
RUN ["python", "download_punkt.py"]
RUN locale-gen en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
RUN ./download_data_ckpt_and_run_inference.sh
CMD ["python", "api/app.py", "--extra_features_top", "--pre_enrich", "--activation_hidden", "--test_batch_size", "1", "--bert_full_embeddings", "--debias_weight", "1.3", "--token_softmax", "--pointer_generator", "--coverage", "--working_dir", "INFERENCE", "--test", "bias_data/WNC/biased.word.test", "--checkpoint", "model.ckpt", "--inference_output", "INFERENCE/output.txt"]
