import streamlit as st
import folium
import requests
import osmnx as ox
import google.generativeai as genai
import speech_recognition as sr
import qrcode
import base64
from streamlit_folium import folium_static
from dotenv import load_dotenv
from twilio.rest import Client
import os

# Load API keys from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE = os.getenv("TWILIO_PHONE")
EMERGENCY_CONTACT = os.getenv("EMERGENCY_CONTACT")

if not GEMINI_API_KEY or not GOOGLE_MAPS_API_KEY:
    st.error("❌ Missing API keys! Please set them in the .env file.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

ox.settings.use_cache = True
ox.settings.log_console = False

def get_coordinates_from_place(place_name):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={place_name}&key={GOOGLE_MAPS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data["results"]:
            return data["results"][0]["geometry"]["location"]["lat"], data["results"][0]["geometry"]["location"]["lng"]
    return None, None

def get_fastest_route(start_lat, start_lon, end_lat, end_lon):
    try:
        osrm_url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
        response = requests.get(osrm_url)
        if response.status_code == 200:
            data = response.json()
            return data["routes"][0]["geometry"]["coordinates"]
    except:
        return None
    return None

def find_nearest_hospitals(lat, lon, search_radius=5000):
    try:
        hospitals = ox.features_from_point((lat, lon), tags={"amenity": "hospital"}, dist=search_radius)
        if hospitals.empty:
            return []
        return [(hospitals.iloc[i]["name"], hospitals.geometry.centroid.iloc[i].y, hospitals.geometry.centroid.iloc[i].x) for i in range(len(hospitals))][:3]
    except:
        return []

def get_best_hospital(hospitals, patient_condition):
    if not hospitals:
        return "No hospitals found"
    prompt = f"Hospitals: {hospitals}\nCondition: {patient_condition}\nBest recommendation?"
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text if response else "No recommendation"

def send_emergency_sms(location):
    if TWILIO_SID and TWILIO_AUTH_TOKEN and EMERGENCY_CONTACT:
        try:
            client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
            message = client.messages.create(
                body=f"🚑 Emergency! Accident at {location}. Immediate assistance required.",
                from_=TWILIO_PHONE,
                to=EMERGENCY_CONTACT
            )
            return message.sid
        except:
            return None
    return None

def generate_qr_code(data):
    qr = qrcode.make(data)
    qr.save("qrcode.png")
    with open("qrcode.png", "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙 Speak now...")
        try:
            audio = recognizer.listen(source, timeout=5)
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Could not recognize speech."
        except sr.RequestError:
            return "Speech recognition service unavailable."
        except Exception:
            return "Error accessing microphone."

st.title("🚑 AI-Based Smart Ambulance Routing System")
accident_location = st.text_input("📍 Enter Accident Location:")
if st.button("🎤 Speak Condition"):
    patient_condition = speech_to_text()
else:
    patient_condition = st.text_area("💉 Enter Patient Condition:")

if accident_location:
    start_lat, start_lon = get_coordinates_from_place(accident_location)
    if start_lat and start_lon:
        hospitals = find_nearest_hospitals(start_lat, start_lon)
        best_hospital = get_best_hospital(hospitals, patient_condition)
        best_hospital_coords = next((h for h in hospitals if h[0] in best_hospital), None)

        if best_hospital_coords:
            route_coordinates = get_fastest_route(start_lat, start_lon, best_hospital_coords[1], best_hospital_coords[2])
            st.subheader("🏥 AI Recommended Hospital:")
            st.write(best_hospital)
            
            if st.button("📢 Send Emergency Alert"):
                sms_status = send_emergency_sms(accident_location)
                if sms_status:
                    st.success("✅ Emergency alert sent successfully!")
                else:
                    st.error("❌ Failed to send emergency alert.")
            
            qr_data = f"Location: {accident_location}, Hospital: {best_hospital}"
            qr_img = generate_qr_code(qr_data)
            st.image(f"data:image/png;base64,{qr_img}", caption="📌 Scan for Location Info")
            
            m = folium.Map(location=[start_lat, start_lon], zoom_start=14)
            folium.Marker([start_lat, start_lon], popup="🚑 Ambulance", icon=folium.Icon(color="red")).add_to(m)
            folium.Marker([best_hospital_coords[1], best_hospital_coords[2]], popup=f"🏥 {best_hospital}", icon=folium.Icon(color="blue")).add_to(m)
            
            if route_coordinates:
                folium.PolyLine([(lat, lon) for lon, lat in route_coordinates], color="blue", weight=5, opacity=0.7).add_to(m)
            folium_static(m)
        else:
            st.error("🚨 No suitable hospital found!")
    else:
        st.error("❌ Invalid location. Please enter a correct place name.")
