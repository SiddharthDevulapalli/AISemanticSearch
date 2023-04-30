# from flask import Flask
# from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from model import SearchModel
import pandas as pd


data_master = pd.read_json('lyrics_200.jl',lines= True)
data = data_master.copy()
print("Data read successfully")

model = SearchModel(data)


print("testing outside search")
user_query = input("enter a query")
print("testing this")

result = model.matching(user_query)
print("Result computed successfully")
output = result.to_json(orient="records")

print(output)

  
# example of another endpoint
# api.add_resource(PredictRatings, '/ratings')

# if __name__ == '__main__':
    # app.run(debug=True)