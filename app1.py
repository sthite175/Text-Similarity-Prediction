###-----------------IMPORT LIBRARIES-----------------------------------------------
from gensim.models import KeyedVectors, Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import pandas as pd
from flask import Flask, request, render_template, jsonify
import utilities

app = Flask(__name__)


# Load Word2Vec Model
word2vec_model = Word2Vec.load("word2vec_model.bin")


@app.route("/")
def Home_Page():
    #return "Welcome To API..."
    return render_template("index.html")

@app.route("/prediction",methods=["POST"])
def text_similarity():
    try:
        # data = request.get_json()
        # text1 = data.get('text1','')
        # text2 = data.get('text2','')

        text1 = request.form.get('text1')
        text2 = request.form.get('text2')


        # DATA PREPROCESSING
        clean_text1 = utilities.remove_blank(text1)
        clean_text2 = utilities.remove_blank(text2)

        clean_text1 = utilities.expand_text(clean_text1)
        clean_text2 = utilities.expand_text(clean_text2)

        clean_text1 = utilities.clean_text(clean_text1)
        clean_text2 = utilities.clean_text(clean_text2)

        clean_text1 = utilities.lemmatization(clean_text1)
        clean_text2 = utilities.lemmatization(clean_text2)

        # Put text data into list in a list
        final_text_1 = [clean_text1]
        final_text_2 = [clean_text2]
        
        final_vector_1 = utilities.vectorizer(final_text_1, word2vec_model)
        final_vector_2 = utilities.vectorizer(final_text_2, word2vec_model)

        vector1 = final_vector_1[0].reshape(1,-1)
        vector2 = final_vector_2[0].reshape(1,-1)

        # COSINE SIMILARITY
        score = cosine_similarity(vector1,vector2)[0][0]
        score = np.round(score,4)
    

        # Return the result in the specified format
        #return jsonify({"similarity score": float(score)})
        return render_template("index.html",result=score)
    
    except Exception as e:
        return render_template("index.html",result=str(e))


if __name__=="__main__":
    app.run(debug=True, port=8080, host="0.0.0.0")

