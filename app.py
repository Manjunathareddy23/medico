import streamlit as st
import folium
import requests
import osmnx as ox
import google.generativeai as genai
from streamlit_folium import folium_static

# Configure Google Gemini API
GEMINI_API_KEY = "your-gemini-api-key"
genai.configure(api_key=GEMINI_API_KEY)

# Ensure OSMnx is properly configured
ox.settings.use_cache = True
ox.settings.log_console = False

# Function to get the fastest route using OSRM API
def get_fastest_route(start_lat, start_lon, end_lat, end_lon):
    try:
        osrm_url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full"
        response = requests.get(osrm_url).json()
        
        if "routes" in response and response["routes"]:
            return response["routes"][0]["geometry"]
        else:
            st.error("No route found. Try different locations.")
            return None
    except Exception as e:
        st.error(f"Error fetching route: {e}")
        return None

# Function to find nearest hospitals using OpenStreetMap
def find_nearest_hospitals(lat, lon, num_hospitals=3, search_radius=5000):
    try:
        # Corrected function call with `dist`
        hospitals = ox.features_from_point((lat, lon), tags={"amenity": "hospital"}, dist=search_radius)

        if hospitals.empty:
            return []

        hospital_list = []
        for _, hospital in hospitals.iterrows():
            name = hospital.get("name", "Unknown Hospital")
            coords = hospital.geometry.centroid
            hospital_list.append((name, coords.y, coords.x))

        return hospital_list[:num_hospitals]

    except Exception as e:
        st.error(f"Error fetching hospitals: {e}")
        return []

# Function to get AI recommendation on best hospital
def get_best_hospital(hospitals, patient_condition):
    if not hospitals:
        return "No recommendation available (No hospitals found)."
    
    prompt = f"""
    Given these hospitals and a patient condition, suggest the best hospital:
    - {hospitals}

    Patient Condition: {patient_condition}

    Consider distance, specialty, and urgency.
    """
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text if response else "No recommendation available."
    
    except Exception as e:
        st.error(f"AI recommendation error: {e}")
        return "No recommendation available."

# Streamlit UI
st.title("🚑 AI-Based Smart Ambulance Routing System")

# User inputs
start_location = st.text_input("Enter Start Location (Latitude, Longitude):")
patient_condition = st.text_area("Enter Patient Condition (e.g., 'Heart Attack, Needs ICU'):")
search_radius = st.slider("Search Radius for Hospitals (meters)", min_value=1000, max_value=10000, value=5000, step=500)

if start_location:
    try:
        start_lat, start_lon = map(float, start_location.split(","))
    except ValueError:
        st.error("Invalid format! Enter coordinates as 'Latitude, Longitude' (e.g., 28.7041, 77.1025).")
        st.stop()

    # Find nearest hospitals
    hospitals = find_nearest_hospitals(start_lat, start_lon, search_radius=search_radius)

    if hospitals:
        st.subheader("🏥 Nearest Hospitals:")
        for idx, (name, lat, lon) in enumerate(hospitals, 1):
            st.write(f"{idx}. {name} - 📍 ({lat}, {lon})")

        # Get AI's best hospital recommendation
        best_hospital = get_best_hospital(hospitals, patient_condition)
        st.subheader("🧠 AI Recommended Hospital:")
        st.write(best_hospital)

        # Get fastest route to the recommended hospital
        best_hospital_coords = next((h for h in hospitals if h[0] in best_hospital), None)

        if best_hospital_coords:
            route_geometry = get_fastest_route(start_lat, start_lon, best_hospital_coords[1], best_hospital_coords[2])

            # Display Map
            m = folium.Map(location=[start_lat, start_lon], zoom_start=14)
            folium.Marker([start_lat, start_lon], popup="🚑 Ambulance", icon=folium.Icon(color="red")).add_to(m)
            folium.Marker([best_hospital_coords[1], best_hospital_coords[2]], popup=f"🏥 {best_hospital}", icon=folium.Icon(color="blue")).add_to(m)
            folium_static(m)
        else:
            st.error("Could not determine the best hospital.")
    else:
        st.error("No hospitals found nearby.")
