FROM unsloth/unsloth

WORKDIR /app

RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    bitsandbytes \
    protobuf \
    sentencepiece

COPY Final-chat.py /app/Final-chat.py

ENV TERM=xterm-256color
ENV TRANSFORMERS_VERBOSITY=error

ENTRYPOINT ["python", "Final-chat.py"]