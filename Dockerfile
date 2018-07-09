FROM pytorch
COPY . .
RUN pip install pandas
ENTRYPOINT python -u train.py > out.txt