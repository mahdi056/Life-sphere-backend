import os
import pickle
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# sustainability model

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "sustainability_model.pkl")

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    features = data.get("features", [])

columns = [
    "Sex", "Age", "Grade",
    "Have_you_taken_an_environmental_education_course",
    "Have_you_taken_a_global_warming_and_climate_course",
    "Have_you_taken_an_Environmental_Literacy_course",
    "Income_status", "Did_you_go_to_pre-school_education",
    "Where_you_live", "Your_number_of_siblings",
] + [f"Q-{i}" for i in range(1, 28)]



# heart disease model

HEART_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "heart_disease.pkl")
with open(HEART_MODEL_PATH, "rb") as f:
    heart_data = pickle.load(f)
    heart_model = heart_data  # SVM model was saved directly
    
# Movie recommendation
MOVIE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "movie_recommender.pkl")

with open(MOVIE_MODEL_PATH, "rb") as f:
    movie_data = pickle.load(f)
    movie_model = movie_data["model"]
    movie_mlb = movie_data["mlb"]        
    movie_onehot = movie_data["onehot"]    
    movie_titles = movie_data["movies"] 

# Crime

CRIME_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "crime.pkl")

with open(CRIME_MODEL_PATH, "rb") as f:
    crime_data = pickle.load(f)

crime_model = crime_data["model"]
crime_encoder = crime_data["encoder"]
cluster_to_risk = crime_data["cluster_to_risk"]

# Pydantic model with mixed types
class InputData(BaseModel):
    Sex: str
    Age: int
    Grade: int
    Have_you_taken_an_environmental_education_course: str
    Have_you_taken_a_global_warming_and_climate_course: str
    Have_you_taken_an_Environmental_Literacy_course: str
    Income_status: int
    Did_you_go_to_pre_school_education: int
    Where_you_live: str
    Your_number_of_siblings: int
    Q_1: str
    Q_2: str
    Q_3: str
    Q_4: str
    Q_5: str
    Q_6: str
    Q_7: str
    Q_8: str
    Q_9: str
    Q_10: str
    Q_11: str
    Q_12: str
    Q_13: str
    Q_14: str
    Q_15: str
    Q_16: str
    Q_17: str
    Q_18: str
    Q_19: str
    Q_20: str
    Q_21: str
    Q_22: str
    Q_23: str
    Q_24: str
    Q_25: str
    Q_26: str
    Q_27: str

    # Added for heart disease prediction
class HeartInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    thalach: float
    exang: float
    oldpeak: float
    ca: float
    thal: float

    # For Movie

class MovieInput(BaseModel):
    genres: str         
    language: str
    country: str
    imdb_score: float

# Crime 

class CrimeInput(BaseModel):
    incident_month: int
    incident_weekday: str
    part_of_the_day: str
    incident_place: str
    incident_district: str
    incident_division: str


@app.post("/predict")
def predict(data: InputData):
    input_dict = data.dict()
    
   
    input_df = pd.DataFrame([{
        col if not col.startswith("Q_") else col.replace("_", "-"): val
        for col, val in input_dict.items()
    }], columns=columns)
    

    numeric_cols = ["Age", "Grade", "Income_status", "Did_you_go_to_pre-school_education", "Your_number_of_siblings"]
    input_df[numeric_cols] = input_df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    
    prediction = model.predict(input_df)[0]
    return {"prediction": int(prediction)}

@app.post("/predict_heart")
def predict_heart(data: HeartInput):
    
    input_data = np.array([[data.age, data.sex, data.cp, data.trestbps,
                            data.chol, data.thalach, data.exang,
                            data.oldpeak, data.ca, data.thal]])
    
    
    prediction = heart_model.predict(input_data)[0]  
    
    return {"prediction": int(prediction)}

@app.post("/recommend_movies")
def recommend_movies(user_input: MovieInput, n: int = 5):
    input_df = pd.DataFrame([user_input.dict()])

    # Wrap single genre into a list for MultiLabelBinarizer
    input_df["genres"] = input_df["genres"].apply(lambda x: [x])

    # Encode genres
    genres_encoded = movie_mlb.transform(input_df["genres"])
    genres_df_input = pd.DataFrame(genres_encoded, columns=movie_mlb.classes_)

    # Encode categorical features
    encoded_cats_input = pd.DataFrame(
        movie_onehot.transform(input_df[["language", "country"]]).toarray()
    )

    # Numerical features
    numerical_input = input_df[["imdb_score"]].reset_index(drop=True)

    # Combine all features
    final_input = pd.concat([numerical_input, genres_df_input, encoded_cats_input], axis=1)

    # Get recommendations
    distances, indices = movie_model.kneighbors(final_input.values)
    recommended = [movie_titles.iloc[indices[0][i]] for i in range(1, n+1)]  

    return {"recommended_movies": recommended}

@app.post("/predict_crime_risk")
def predict_crime_risk(user_input: CrimeInput):
    input_dict = user_input.dict()

    # Do NOT pop month; keep it with other features
    df = pd.DataFrame([input_dict])  # contains month + strings

    # Encode everything
    encoded = crime_encoder.transform(df)

    # Use encoded data for prediction
    cluster = crime_model.predict(encoded)[0]
    risk = cluster_to_risk.get(cluster, "Unknown")

    return {"cluster": int(cluster), "risk_level": risk}


@app.get("/")
def root():
    return {"message": "FastAPI backend is running."}
