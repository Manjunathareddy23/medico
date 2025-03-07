import streamlit as st
import folium
import requests
import osmnx as ox
import google.generativeai as genai
from streamlit_folium import folium_static

# Configure Google Gemini API
GEMINI_API_KEY = "your-gemini-api-key"
genai.configure(api_key=GEMINI_API_KEY)

# Function to get the fastest route using OSRM API
def get_fastest_route(start_lat, start_lon, end_lat, end_lon):
    osrm_url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full"
    response = requests.get(osrm_url).json()
    
    if response.get("routes"):
        return response["routes"][0]["geometry"]
    return None

# Function to find nearest hospitals using OpenStreetMap
def find_nearest_hospitals(lat, lon, num_hospitals=3):
    hospitals = ox.geometries_from_point((lat, lon), tags={"amenity": "hospital"})
    hospital_list = []
    for _, hospital in hospitals.iterrows():
        name = hospital.get("name", "Unknown Hospital")
        coords = hospital.geometry.centroid
        hospital_list.append((name, coords.y, coords.x))
    return hospital_list[:num_hospitals]

# Function to get AI recommendation on best hospital
def get_best_hospital(hospitals, patient_condition):
    prompt = f"""
    Given these hospitals and a patient condition, suggest the best hospital:
    - {hospitals}

    Patient Condition: {patient_condition}

    Consider distance, specialty, and urgency.
    """
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text if response else "No recommendation available."

# Streamlit UI
st.title("🚑 AI-Based Smart Ambulance Routing System")

# User inputs
start_location = st.text_input("Enter Start Location (Latitude, Longitude):")
patient_condition = st.text_area("Enter Patient Condition (e.g., 'Heart Attack, Needs ICU'):")
if start_location:
    start_lat, start_lon = map(float, start_location.split(","))
    
    # Find nearest hospitals
    hospitals = find_nearest_hospitals(start_lat, start_lon)
    
    if hospitals:
        st.subheader("🏥 Nearest Hospitals:")
        for idx, (name, lat, lon) in enumerate(hospitals, 1):
            st.write(f"{idx}. {name} - 📍 ({lat}, {lon})")

        # Get AI's best hospital recommendation
        best_hospital = get_best_hospital(hospitals, patient_condition)
        st.subheader("🧠 AI Recommended Hospital:")
        st.write(best_hospital)

        # Get fastest route to the recommended hospital
        best_hospital_coords = next(h for h in hospitals if h[0] in best_hospital)
        route_geometry = get_fastest_route(start_lat, start_lon, best_hospital_coords[1], best_hospital_coords[2])

        # Display Map
        m = folium.Map(location=[start_lat, start_lon], zoom_start=14)
        folium.Marker([start_lat, start_lon], popup="🚑 Ambulance", icon=folium.Icon(color="red")).add_to(m)
        folium.Marker([best_hospital_coords[1], best_hospital_coords[2]], popup=f"🏥 {best_hospital}", icon=folium.Icon(color="blue")).add_to(m)
        folium_static(m)
    else:
        st.error("No hospitals found nearby.")
