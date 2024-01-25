# Project : Text Similarity Prediction

## Problem Statement:
Develop a model to measure the semantic similarity between pairs of text paragraphs, predicting a score from 0 (highly dissimilar) to 1 (highly similar). Deploy this model as a server API endpoint on a cloud service provider for practical accessibility and usability.

## Introduction :
The Text Similarity Prediction project aims to develop a system that quantifies the semantic similarity between two text inputs. The project utilizes Natural Language Processing (NLP)  techniques, including word embeddings and cosine similarity, to measure the semantic equivalence of two given texts. The primary objective is to provide users with a tool that can assess the degree of similarity between sentences or paragraphs.

## Project Pipeline :
1.Problem Statement
2.Data Collection
3.Data Preprocessing
4.Feature Engineering
5.Model Building
6.Create API
7.Containerization Code Using Docker
8.Github Repository
9.Deployment on Cloud Platform 


## Tools and Libraries Use :
1.Programming Language : Python
2.Machine Learning Library: sklearn, Numpy , Pandas
3.NLP Tool : NLTK, Gensim
4.Web Framework : Flask
5.Integrated Development Environment (IDE): vscode, Jupyter Notebook
6.Version Control : Github
7.Cloud Service : Render


# Project Components:

## Data Preprocessing : 
We have text data we load data and start text data preprocessing which contains tokenization, text cleaning, stemming and lemmatization and so on.

## Steps in Text Preprocessing:
1. Tokenization
2. Remove Blank Spaces
3. Handling Accented Characters
4. Expanding Text
5. Remove Stopwords, Punctuation, Numeric and Symbols
6. Stemming and Lemmatization

## Feature Engineering :
It is the process of extracting meaningful information from raw data to make it usable for NLP models. We have to convert text into numerical form , we use word embedding techniques like TFIDF, CountVectorizer, Word2vec, FastText, OpenAIEmbedding, BERT etc.

## Word2Vec : 
In this project we use word2vec word embedding technique.
It is pre-trained model build on simple neural network. It convert text data into vector representation form which having meaningful information.

## Cosine Similarity :
Cosine similarity is employed to measure the similarity
between two vectors representing the text inputs. The cosine similarity function from scikit-learn is utilized to compute the cosine similarity score, indicating the degree of semantic similarity between the two texts.
 

## Model Building :
We build model by using  Word2Vec embedding. When user pass two text input our model will predict as similarity score between two text.

## Flask API :
The project is deployed as a web application using Flask, a lightweight Python web framework. The app.py file defines routes for the home page and the text similarity prediction endpoint. Users can input two text paragraphs through a web form, and the application returns the calculated similarity score.


## Docker Containerization:
The project is containerized using Docker to ensure consistent and reproducible environments. The Dockerfile specifies the base image, installs necessary dependencies from the requirements.txt file, and sets up the environment for running the Flask application.

## Github :
Create repository and push code on this repository and try to deploy by using this repository on free cloud platform Render.


## Deployment on Cloud Platform:

We deploy our project on AWS platform and also deploy on Render platform.



## Conclusion:

The Text Similarity Prediction project successfully implements a system for quantifying semantic similarity between two text inputs. It leverages word embeddings, text preprocessing techniques, and cosine similarity to achieve accurate similarity predictions. The user-friendly web interface allows easy interaction with the system, making it a practical tool for various applications requiring text similarity assessments. Text Similarity Prediction project provides a valuable solution for users seeking to evaluate the semantic equivalence of text inputs
## Run Locally

Clone the project

```bash
  git clone https://github.com/sthite175/Text-Similarity-Prediction
```

Go to the project directory

```bash
  cd Text-Similarity-Prediction
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  python app1.py
```


## Authors

- [@sunilthite](https://www.github.com/sthite175)


## 🚀 About Me
I am Associate AI Engineer having more than 3.3 years of experience. I am expertise in machine learning, deep learning and NLP to develop innovative AI solutions that solve complex business problems and improve customer experience. Skilled in python and SQL with strong understanding of ML libraries and data visualization techniques such as sklearn, matplotlib and seaborn. I am passionate about staying up-to-date with the latest developments in the field.

# My Portfolio

## Kaggle
- [Link to my Kaggle profile](https://www.kaggle.com/sunilthite)

## GitHub
- [Link to my GitHub profile](https://github.com/sthite175)

## LinkedIn Account
- [Link to my account page](https://www.linkedin.com/in/sunil-thite-a04745271)






## 🛠 Skills
Python, SQL, HTML, Numpy, Pandas, Machine Learning, Deep Learning, NLP, Data Visualization, matplotlib, seaborn, sklearn, nltk, tensorflow, ANN, CNN, RNN, LSTM, Docker, Kubernetes, github


## Support

For support, email sthite175@gmail.com

