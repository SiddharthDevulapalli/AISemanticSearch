from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from model import SearchModel
import pandas as pd

app = Flask(__name__)
api = Api(app)
# create new model object

# load trained classifier
# data_path = 'lib/models/data.pkl'
# with open(clf_path, 'rb') as f:
    # model.clf = pickle.load(f)
# load trained vectorizer
# vec_path = 'lib/models/TFIDFVectorizer.pkl'
# with open(vec_path, 'rb') as f:
    # model.vectorizer = pickle.load(f)

data_master = pd.read_json('lyrics_200.jl',lines= True)
data = data_master.copy()
print("Data read successfully")

model = SearchModel(data)

parser = reqparse.RequestParser()
parser.add_argument('query')
print("testing outside search")
class Search(Resource):
    def get(self):
        print("testing this")
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']

        # vectorize the user's query and make a prediction
        # uq_vectorized = model.vectorizer_transform(
        #     np.array([user_query]))
        # prediction = model.predict(uq_vectorized)
        # pred_proba = model.predict_proba(uq_vectorized)
        # Output 'Negative' or 'Positive' along with the score
        # if prediction == 0:
        #     pred_text = 'Negative'
        # else:
        #     pred_text = 'Positive'
            
        # round the predict proba value and set to new variable
        # confidence = round(pred_proba[0], 3)
        # create JSON object
        # output = {'prediction': pred_text, 'confidence': confidence}
        result = model.matching(user_query)
        print("Result computed successfully")
        output = result.to_json(orient="records")
        return result

api.add_resource(Search, '/')
  
# example of another endpoint
# api.add_resource(PredictRatings, '/ratings')

if __name__ == '__main__':
    app.run(debug=True)