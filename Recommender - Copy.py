import sklearn
import pickle
import numpy as np
import pandas as pd
import ast
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation
model = pickle.load(open('Descision_tree.pkl','rb'))
ratings = pd.read_csv('808ratings.csv')
users = pd.read_csv('808users.csv')
user_ratings = pd.merge(users, ratings)
def predict ():
    print("Answer the below questions ranking 1-5. where\n 1-strongly negative\n2-negative\n3-neutral\n4-positive\n5-strongly positive")
    print("\nWhat time do you generally wake up  \n (1 - Before 5:00 AM , 2 - Between 5:00 AM  to 6:00 AM ,3 - Between  6:00 AM - 7:00 AM,4 - Between  7:00 AM - 8:00 AM,5 - after 8:00AM :")
    input1 = int(input())
    print("\nWhat time do you go to bed? \n (1 - before 9:00 PM , 2 - before 10:00 PM ,3 - before 11:00 PM,4 - before 12:00 PM,5 - after 12:00 PM:")
    input2 = int(input())
    print("\n How neat or messy are you?(1-5): ")
    input3 = int(input())
    print("\nRate yourself as a Conservative Person: ")
    input4 = int(input())
    print("\nwhat type of food you like?: 1.Veg 2.Non-veg 3.Both veg and Non-veg 4.High Calorie food 5.Doesn't Matter")
    input5 = int(input())
    print("\n How often do you travel?(1-5): ")
    input6 = int(input())
    print("\nDo you like doing parties:? ")
    input7 = int(input())
    print("\nRate yourself as a introvert: ")
    input8 = int(input())
    print("\n Are you 1. North Indian 2. South Indian 3. North East 4.Asian 5.Outside Of Asia:")
    input9 = int(input())
    print("\nRate yourself as a Extrovert: ")
    input10 = int(input())
    print("\nwhat is your mother tongue?: 1.Hindi 2.Tamil 3.Telgu 4.Marathi 5.English")
    input11 = int(input())
    prediction = model.predict([[input1,input2,input3,input4,input5,input6,input7,input8,input9,input10,input11]])
    prediction = str(prediction).lstrip('[').rstrip(']')
    prediction = prediction.strip('')
    result = ast.literal_eval(prediction)
    print(result)
    ratings_matrix = ratings.pivot_table(index=['user_id'],columns=['ext_id'],values='rating')
    ratings_matrix.fillna( 0, inplace = True )

    #Calculating Cosine Similiarities
    user_similarity = 1 - pairwise_distances( ratings_matrix.to_numpy(), metric="cosine" )

    #Filling diagonals with 0s instead of 1 to avoid recommending the same user when sorting
    np.fill_diagonal( user_similarity, 0 )
    ratings_matrix = pd.DataFrame( user_similarity )

    try:
        user_inp=result
        inp = users[users['title'] == user_inp].index.tolist()
        inp = inp[0]

        users['similarity'] = ratings_matrix.iloc[inp]
        users.columns = ['user_id', 'title', 'similarity']
        print("Recommended users based on", user_inp, ": \n",users.sort_values(["similarity"], ascending=False)[1:10])

    except:
        print("Sorry, the user is not in the database!")

predict()
