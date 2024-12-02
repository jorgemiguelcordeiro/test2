'''
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load the saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Feature names in the correct order
feature_names = [
    'BMI', 'Age', 'HighBP', 'HighChol', 'GenHlth',
    'PhysHlth', 'MentHlth', 'Smoker', 'PhysActivity', 'Sex'
]

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form = await request.form()
    data = [float(form[feature]) for feature in feature_names]
    input_data = np.array([data])
    input_data_scaled = scaler.transform(input_data)
    y_pred_prob = model.predict_proba(input_data_scaled)[:, 1]
    threshold = 0.3
    y_pred_thresh = (y_pred_prob >= threshold).astype(int)
    result = "The patient is likely to have diabetes." if y_pred_thresh[0] == 1 else "The patient is not likely to have diabetes."
    return templates.TemplateResponse("result.html", {
        "request": request,
        "result": result,
        "probability": y_pred_prob[0]
    })
'''

#with fast api

'''
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load the saved model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Feature names in the correct order
feature_names = ['BMI', 'Age', 'HighBP', 'HighChol', 'GenHlth', 'PhysHlth', 'MentHlth', 'Smoker', 'PhysActivity', 'Sex']

# Onboarding page (appears first)
@app.get("/", response_class=HTMLResponse)
async def onboarding(request: Request):
    return templates.TemplateResponse("Onboarding.html", {"request": request})

# Redirect to Home after onboarding
# Redirect to Home after onboarding
@app.post("/start")
async def start():
    return RedirectResponse(url="/home", status_code=303)
from fastapi import Form

@app.post("/save_tracking_data")
async def save_tracking_data(request: Request, sleep: float = Form(...), steps: int = Form(...)):
    # Process the data here (e.g., save it or log it for demonstration)
    print(f"Sleep hours: {sleep}, Steps walked: {steps}")

    # Redirect back to home or another relevant page after saving the data
    return RedirectResponse(url="/home", status_code=303)


# Home page with navigation to all features
@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Tracking page
@app.get("/tracking", response_class=HTMLResponse)
async def tracking(request: Request):
    return templates.TemplateResponse("Tracking.html", {"request": request})

# Gamification page
@app.get("/gamification", response_class=HTMLResponse)
async def gamification(request: Request):
    return templates.TemplateResponse("gamification.html", {"request": request})

# Insights page
@app.get("/insights", response_class=HTMLResponse)
async def insights(request: Request):
    return templates.TemplateResponse("Insights.html", {"request": request})

# Prediction form page
@app.get("/predict_form", response_class=HTMLResponse)
async def predict_form(request: Request):
    return templates.TemplateResponse("predict_form.html", {"request": request})

@app.get("/redeem_rewards", response_class=HTMLResponse)
async def redeem_rewards(request: Request):
    return templates.TemplateResponse("redeem_rewards.html", {"request": request})


# Predict result
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form_data = await request.form()
    try:
        data = [float(form_data[feature]) for feature in feature_names]
    except KeyError as e:
        return HTMLResponse(content=f"Missing field: {str(e)}", status_code=400)
    except ValueError as e:
        return HTMLResponse(content=f"Invalid value: {str(e)}", status_code=400)

    # Prepare and scale input data
    input_data = np.array([data])
    input_data_scaled = scaler.transform(input_data)
    y_pred_prob = model.predict_proba(input_data_scaled)[:, 1]
    probability = y_pred_prob[0] * 100  # Convert to percentage for easier interpretation

    # Multi-level threshold evaluation
    if probability >= 75:
        result = "The patient has a high likelihood of having diabetes."
    elif 50 <= probability < 75:
        result = "The patient has a moderate likelihood of having diabetes."
    elif 30 <= probability < 50:
        result = "The patient has a low likelihood of having diabetes."
    else:
        result = "The patient is unlikely to have diabetes."

    # Return result and exact probability
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "result": result,
            "probability": probability,  # Exact probability, not rounded
        },
    )
'''

import streamlit as st
import numpy as np
import pickle

# Load the saved model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Feature names in the correct order
feature_names = ['BMI', 'Age', 'HighBP', 'HighChol', 'GenHlth', 'PhysHlth', 'MentHlth', 'Smoker', 'PhysActivity', 'Sex']

# Form to collect data
st.title("Diabetes Prediction")
data = []
for feature in feature_names:
    # Use an appropriate input widget based on data type
    data.append(st.number_input(f'Enter {feature}', format="%.2f"))

# Button to make prediction
if st.button('Predict'):
    try:
        input_data = np.array([data])
        input_data_scaled = scaler.transform(input_data)
        y_pred_prob = model.predict_proba(input_data_scaled)[:, 1]
        probability = y_pred_prob[0] * 100  # Convert to percentage

        # Multi-level threshold evaluation
        if probability >= 75:
            result = "The patient has a high likelihood of having diabetes."
        elif 50 <= probability < 75:
            result = "The patient has a moderate likelihood of having diabetes."
        elif 30 <= probability < 50:
            result = "The patient has a low likelihood of having diabetes."
        else:
            result = "The patient is unlikely to have diabetes."

        st.write(f'Result: {result}')
        st.write(f'Probability of Diabetes: {probability:.2f}%')
    except Exception as e:
        st.error(f"Error: {str(e)}")

