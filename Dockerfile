FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/workspace/.torch

RUN pip install --no-cache-dir opencv-python-headless==4.10.0.84 \
        scikit-learn==1.5.1 pandas==2.2.2 tabulate==0.9.0

WORKDIR /app
COPY *.py run_all.sh /app/
RUN chmod +x /app/run_all.sh

CMD ["bash", "/app/run_all.sh"]
