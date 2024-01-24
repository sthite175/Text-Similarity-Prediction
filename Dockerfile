# From base image
FROM python:3.10

COPY . /home
WORKDIR /home

# upgrade pip  and install library from requirements.txt 
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download all NLTK data
RUN python -c "import nltk; nltk.download('stopwords')"
RUN python -c "import nltk; nltk.download('wordnet')"


EXPOSE 8080

CMD  ["python", "app1.py"]