import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# Load the model
def load_model(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

model = load_model("./data/models/csgo_model.pkl")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow POST, GET, OPTIONS, etc.
    allow_headers=["*"],  # Allow all headers
)

# Define input schema
class MatchData(BaseModel):
    time_left: int
    ct_score: int
    t_score: int
    map: str
    bomb_planted: bool
    ct_defuse_kits: int
    ct_health: int
    t_health: int
    ct_armor: int
    t_armor: int
    ct_helmets: int
    t_helmets: int
    ct_money: int
    t_money: int
    ct_players_alive: int
    t_players_alive: int
    ct_weapons: list[str]
    t_weapons: list[str]

# List of all possible features
ALL_FEATURES = [
    "time_left",
    "ct_score",
    "t_score",
    "bomb_planted",
    "ct_health",
    "t_health",
    "ct_armor",
    "t_armor",
    "ct_money",
    "t_money",
    "ct_helmets",
    "t_helmets",
    "ct_defuse_kits",
    "ct_players_alive",
    "t_players_alive",
    "ct_weapon_ak47",
    "t_weapon_ak47",
    "ct_weapon_aug",
    "t_weapon_aug",
    "ct_weapon_awp",
    "t_weapon_awp",
    "ct_weapon_cz75auto",
    "t_weapon_cz75auto",
    "ct_weapon_famas",
    "t_weapon_famas",
    "ct_weapon_galilar",
    "t_weapon_galilar",
    "ct_weapon_glock",
    "t_weapon_glock",
    "ct_weapon_m4a1s",
    "ct_weapon_m4a4",
    "t_weapon_m4a4",
    "ct_weapon_mac10",
    "t_weapon_mac10",
    "ct_weapon_mag7",
    "ct_weapon_mp9",
    "t_weapon_mp9",
    "ct_weapon_sg553",
    "t_weapon_sg553",
    "ct_weapon_ssg08",
    "t_weapon_ssg08",
    "ct_weapon_ump45",
    "t_weapon_ump45",
    "ct_weapon_xm1014",
    "ct_weapon_deagle",
    "t_weapon_deagle",
    "ct_weapon_fiveseven",
    "t_weapon_fiveseven",
    "ct_weapon_usps",
    "t_weapon_usps",
    "ct_weapon_p250",
    "t_weapon_p250",
    "ct_weapon_p2000",
    "t_weapon_p2000",
    "ct_weapon_tec9",
    "t_weapon_tec9",
    "ct_grenade_hegrenade",
    "t_grenade_hegrenade",
    "ct_grenade_flashbang",
    "t_grenade_flashbang",
    "ct_grenade_smokegrenade",
    "t_grenade_smokegrenade",
    "ct_grenade_incendiarygrenade",
    "t_grenade_incendiarygrenade",
    "ct_grenade_molotovgrenade",
    "t_grenade_molotovgrenade",
    "ct_grenade_decoygrenade",
    "t_grenade_decoygrenade",
    "de_dust2",
    "de_inferno",
    "de_mirage",
    "de_nuke",
    "de_overpass",
    "de_train",
    "de_vertigo"
]


def preprocess_input(data: MatchData):
    # Initialize feature vector
    feature_vector = [0] * len(ALL_FEATURES)

    # Map simple fields
    feature_vector[ALL_FEATURES.index("time_left")] = data.time_left
    feature_vector[ALL_FEATURES.index("ct_score")] = data.ct_score
    feature_vector[ALL_FEATURES.index("t_score")] = data.t_score
    feature_vector[ALL_FEATURES.index("bomb_planted")] = int(data.bomb_planted)
    feature_vector[ALL_FEATURES.index("ct_health")] = data.ct_health
    feature_vector[ALL_FEATURES.index("t_health")] = data.t_health
    feature_vector[ALL_FEATURES.index("ct_armor")] = data.ct_armor
    feature_vector[ALL_FEATURES.index("t_armor")] = data.t_armor
    feature_vector[ALL_FEATURES.index("ct_money")] = data.ct_money
    feature_vector[ALL_FEATURES.index("t_money")] = data.t_money
    feature_vector[ALL_FEATURES.index("ct_helmets")] = data.ct_helmets
    feature_vector[ALL_FEATURES.index("t_helmets")] = data.t_helmets
    feature_vector[ALL_FEATURES.index("ct_defuse_kits")] = data.ct_defuse_kits
    feature_vector[ALL_FEATURES.index("ct_players_alive")] = data.ct_players_alive
    feature_vector[ALL_FEATURES.index("t_players_alive")] = data.t_players_alive

    # Map weapons
    for weapon in data.ct_weapons:
        if weapon in ALL_FEATURES:
            feature_vector[ALL_FEATURES.index(weapon)] += 1
    for weapon in data.t_weapons:
        if weapon in ALL_FEATURES:
            feature_vector[ALL_FEATURES.index(weapon)] += 1

    # Map the selected map
    map_feature = f"{data.map}"
    if map_feature in ALL_FEATURES:
        feature_vector[ALL_FEATURES.index(map_feature)] = 1
    else:
        # All maps = 0
        feature_vector[ALL_FEATURES.index("map_de_dust2")] = 0
        feature_vector[ALL_FEATURES.index("map_de_inferno")] = 0
        feature_vector[ALL_FEATURES.index("map_de_mirage")] = 0
        feature_vector[ALL_FEATURES.index("map_de_vertigo")] = 0
        feature_vector[ALL_FEATURES.index("map_de_nuke")] = 0
        feature_vector[ALL_FEATURES.index("map_de_overpass")] = 0
        feature_vector[ALL_FEATURES.index("map_de_train")] = 0

    return feature_vector

@app.post("/predict")
async def predict_endpoint(match_data: MatchData):
    # Preprocess input
    features = preprocess_input(match_data)
    
    # Create DataFrame properly by making features a single row
    featuresdf = pd.DataFrame([features], columns=ALL_FEATURES)
    
    # Make prediction and convert numpy.int64 to regular Python int
    prediction = model.predict(featuresdf)
    prediction_value = int(prediction[0])  # Convert numpy.int64 to Python int
    
    if prediction_value == 1:
        prediction_value = "CT"
    else:
        prediction_value = "T"
    return {"prediction": prediction_value}