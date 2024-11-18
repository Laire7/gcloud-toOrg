FROM python:3.9-slim

WORKDIR /app

#COPY requirements.txt .
RUN pip install streamlit

COPY . .
ENTRYPOINT ["streamlit","run"]
CMD ["app.py","--server.enableCORS","false","--browser.serverAddress","0.0.0.0","--browser.gatherUsageStats","false","--server.port","8080"]