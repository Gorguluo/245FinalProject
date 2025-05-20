
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


st.write("Adjust the input values below to simulate a match and predict whether the Radiant team will win.")

game_mode = st.selectbox("Game Mode", list(range(1, 25)), index=0)
lobby_type = st.selectbox("Lobby Type", list(range(0, 7)), index=0)
game_time = st.slider("Game Time (seconds)", 0, 4000, 1500)

st.subheader("Radiant Team Stats")
radiant_gold = st.slider("Total Radiant Gold", 0, 100000, 25000)
radiant_xp = st.slider("Total Radiant XP", 0, 100000, 25000)
radiant_kda = st.slider("Average Radiant KDA", 0.0, 15.0, 4.0)

st.subheader("Dire Team Stats")
dire_gold = st.slider("Total Dire Gold", 0, 100000, 25000)
dire_xp = st.slider("Total Dire XP", 0, 100000, 25000)
dire_kda = st.slider("Average Dire KDA", 0.0, 15.0, 4.0)

input_data = pd.DataFrame([{
    'game_mode': game_mode,
    'lobby_type': lobby_type,
    'game_time': game_time,
    'radiant_total_gold': radiant_gold,
    'dire_total_gold': dire_gold,
    'radiant_total_xp': radiant_xp,
    'dire_total_xp': dire_xp,
    'radiant_kda': radiant_kda,
    'dire_kda': dire_kda
}])


try:
    train = pd.read_csv("train_features.csv")
    targets = pd.read_csv("train_targets.csv")
    df = pd.merge(train, targets[['match_id_hash', 'radiant_win']], on='match_id_hash')
    df = df.sample(n=10000, random_state=42)

    df['radiant_total_gold'] = df[[col for col in df.columns if col.startswith('r') and col.endswith('_gold')]].sum(axis=1)
    df['dire_total_gold'] = df[[col for col in df.columns if col.startswith('d') and col.endswith('_gold')]].sum(axis=1)
    df['radiant_total_xp'] = df[[col for col in df.columns if col.startswith('r') and col.endswith('_xp')]].sum(axis=1)
    df['dire_total_xp'] = df[[col for col in df.columns if col.startswith('d') and col.endswith('_xp')]].sum(axis=1)

    radiant_kills = df[[col for col in df.columns if col.startswith('r') and col.endswith('_kills')]].sum(axis=1)
    radiant_assists = df[[col for col in df.columns if col.startswith('r') and col.endswith('_assists')]].sum(axis=1)
    radiant_deaths = df[[col for col in df.columns if col.startswith('r') and col.endswith('_deaths')]].replace(0, 1)
    df['radiant_kda'] = (radiant_kills + radiant_assists) / radiant_deaths

    dire_kills = df[[col for col in df.columns if col.startswith('d') and col.endswith('_kills')]].sum(axis=1)
    dire_assists = df[[col for col in df.columns if col.startswith('d') and col.endswith('_assists')]].sum(axis=1)
    dire_deaths = df[[col for col in df.columns if col.startswith('d') and col.endswith('_deaths')]].replace(0, 1)
    df['dire_kda'] = (dire_kills + dire_assists) / dire_deaths

    X = df[['game_mode', 'lobby_type', 'game_time', 'radiant_total_gold', 'dire_total_gold',
            'radiant_total_xp', 'dire_total_xp', 'radiant_kda', 'dire_kda']]
    y = df['radiant_win']

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_imputed, y)

    input_imputed = imputer.transform(input_data)
    prediction = model.predict(input_imputed)[0]
    confidence = model.predict_proba(input_imputed)[0][prediction]

    st.subheader("Prediction;")
    if prediction == 1:
        st.success(f"Radiant is predicted to **WIN** with {confidence:.2%} confidence.")
    else:
        st.error(f"Radiant is predicted to **LOSE** with {confidence:.2%} confidence.")
except Exception as e:
    st.error(f"An error occurred: {e}")
