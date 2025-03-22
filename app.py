import numpy as np
import tensorflow as tf
import pickle
import gradio as gr
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load trained model
# Load trained model
model = tf.keras.models.load_model("model.h5",compile=True)

# Fix: Compile the model after loading to build the missing metrics
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Save the trained model
model.save("model.h5", save_format="h5")


# Load Label Encoder safely
try:
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    if not isinstance(encoder, LabelEncoder) or not hasattr(encoder, "classes_"):
        print("⚠ Warning: encoder.pkl is not valid. Re-fitting now.")
        encoder = LabelEncoder()
except:
    print("⚠ Warning: Failed to load encoder.pkl. Creating a new LabelEncoder.")
    encoder = LabelEncoder()

# Load dataset
train_df = pd.read_csv("Training.csv")

# Identify disease column
for col in train_df.columns:
    if col.strip().lower() == "prognosis":
        disease_col = col
        break
else:
    raise ValueError("⚠ Error: 'Disease' column not found in Training.csv. Please check column names.")

# Fit LabelEncoder
encoder.fit(train_df[disease_col])

# Save corrected encoder
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Load MinMaxScaler safely
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    if not isinstance(scaler, MinMaxScaler) or not hasattr(scaler, "data_min_"):
        print("⚠ Warning: scaler.pkl is invalid. Re-fitting now.")
        scaler = MinMaxScaler()
except:
    print("⚠ Warning: Failed to load scaler.pkl. Creating a new MinMaxScaler.")
    scaler = MinMaxScaler()

# Fit scaler if necessary
X_train = train_df.iloc[:, :-1].values  
if not hasattr(scaler, "data_min_"):
    print("🔄 Fitting MinMaxScaler on training data...")
    scaler.fit(X_train)

# Save the fitted scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Load additional disease information
data_df = pd.read_csv("new_data3 (3).csv")

# Get symptom columns
symptom_columns = train_df.columns[:-1]

def transform_input(user_symptoms):
    input_vector = np.zeros(len(symptom_columns))
    for i, col in enumerate(symptom_columns):
        if col.lower() in user_symptoms:
            input_vector[i] = 1  
    return scaler.transform([input_vector])

def get_disease_info(disease_name):
    details = data_df[data_df['Disease'] == disease_name].iloc[0] if not data_df[data_df['Disease'] == disease_name].empty else None
    
    return {
        "description": details["description"] if details is not None else "No description available.",
        "medication": eval(details["medication"]) if details is not None else [],
        "diet": eval(details["diet"]) if details is not None else [],
        "precautions": [details["Precaution_1"], details["Precaution_2"], details["Precaution_3"], details["Precaution_4"]] if details is not None else [],
        "workout_advice": details["Grouped_Workout_Advice"] if details is not None else "No advice available."
    }

def get_top_diseases(symptoms):
    input_data = transform_input(symptoms)
    prediction_probs = model.predict(input_data)[0]
    top_3_indices = prediction_probs.argsort()[-3:][::-1]
    top_3_diseases = encoder.inverse_transform(top_3_indices)
    top_3_probs = prediction_probs[top_3_indices]
    return list(zip(top_3_diseases, top_3_probs))

def chatbot_response(user_input):
    try:
        symptoms = [sym.strip().lower() for sym in user_input.split(",")]
        predictions = get_top_diseases(symptoms)
        response = "🔎 *Predicted Diseases:*\n\n"


        for disease, prob in predictions:
            info = get_disease_info(disease)
            response += f"🔹 *{disease}* ({prob:.2f} confidence)\n\n"

            response += f"📌 *Description:* {info['description']}\n\n"

            response += f"💊 *Medications:* {', '.join(info['medication'])if info['medication'] else 'No medication available.'}\n\n"

            response += f"🥗 *Recommended Diet:* {', '.join(info['diet'])}\n\n"

            response += f"⚠ *Precautions:* {', '.join(info['precautions'])}\n\n"

            response += f"🏋 *Workout Advice:* {info['workout_advice']}\n\n"


        return response
    except Exception as e:
        return f"⚠ Error: {str(e)}"

iface = gr.Interface(fn=chatbot_response, inputs="text", outputs="text",
                     title="Disease Prediction Chatbot",
                     description="Enter symptoms separated by commas (e.g.,irritation,skin_rash,nodal_skin_eruptions,cough,high_fever,runny_nose,etc.) to get disease predictions.")

iface.launch()
