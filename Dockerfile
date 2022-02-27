FROM python:3.8
COPY . .
RUN pip install -r requirements.txt
RUN pip install catboost
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]
