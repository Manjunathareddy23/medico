import streamlit as st
import folium
import requests
import osmnx as ox
import google.generativeai as genai
from streamlit_folium import folium_static
from dotenv import load_dotenv
import os

# Load API keys from .env file
load_dotenv()

# Configure Google Gemini API securely
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")  # Optional: Use Google Maps for geocoding

if not GEMINI_API_KEY:
    st.error("❌ API key not found! Please set GEMINI_API_KEY in the .env file.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Ensure OSMnx is properly configured
ox.settings.use_cache = True
ox.settings.log_console = False

# Function to convert place name to coordinates using Nominatim API or Google Maps API
def get_coordinates_from_place(place_name):
    try:
        if GOOGLE_MAPS_API_KEY:  # Use Google Maps API if available
            url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {"address": place_name, "key": GOOGLE_MAPS_API_KEY}
        else:  # Fallback to OpenStreetMap's Nominatim API
            url = "https://nominatim.openstreetmap.org/search"
            params = {"q": place_name, "format": "json"}
            headers = {"User-Agent": "Streamlit-Ambulance-App"}

        response = requests.get(url, params=params).json()

        if response and "results" in response:  # Google Maps response
            lat, lon = response["results"][0]["geometry"]["location"].values()
        elif response:  # Nominatim response
            lat, lon = float(response[0]["lat"]), float(response[0]["lon"])
        else:
            st.error("❌ Could not find location. Try entering a more specific place.")
            return None, None

        return lat, lon
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error: {e}")
        return None, None
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None, None

# Function to get the fastest route using OSRM API
def get_fastest_route(start_lat, start_lon, end_lat, end_lon):
    try:
        osrm_url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
        response = requests.get(osrm_url).json()

        if "routes" in response and response["routes"]:
            return response["routes"][0]["geometry"]["coordinates"]
        else:
            st.error("❌ No route found. Try different locations.")
            return None
    except Exception as e:
        st.error(f"❌ Error fetching route: {e}")
        return None

# Function to find nearest hospitals using OpenStreetMap
def find_nearest_hospitals(lat, lon, num_hospitals=3, search_radius=5000):
    try:
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
        st.error(f"❌ Error fetching hospitals: {e}")
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
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text if response else "No recommendation available."
    
    except Exception as e:
        st.error(f"❌ AI recommendation error: {e}")
        return "No recommendation available."

# Streamlit UI
st.title("🚑 AI-Based Smart Ambulance Routing System")

# User inputs
accident_location = st.text_input("📍 Enter Accident Location (e.g., Connaught Place, New Delhi):")
patient_condition = st.text_area("💉 Enter Patient Condition (e.g., 'Heart Attack, Needs ICU'):")
search_radius = st.slider("📡 Search Radius for Hospitals (meters)", min_value=1000, max_value=10000, value=5000, step=500)

if accident_location:
    # Convert location name to latitude and longitude
    start_lat, start_lon = get_coordinates_from_place(accident_location)

    if start_lat is None or start_lon is None:
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
            route_coordinates = get_fastest_route(start_lat, start_lon, best_hospital_coords[1], best_hospital_coords[2])

            # Display Map
            m = folium.Map(location=[start_lat, start_lon], zoom_start=14)
            folium.Marker([start_lat, start_lon], popup="🚑 Ambulance", icon=folium.Icon(color="red")).add_to(m)
            folium.Marker([best_hospital_coords[1], best_hospital_coords[2]], popup=f"🏥 {best_hospital}", icon=folium.Icon(color="blue")).add_to(m)

            # Draw route if available
            if route_coordinates:
                folium.PolyLine([(lat, lon) for lon, lat in route_coordinates], color="blue", weight=5, opacity=0.7).add_to(m)

            folium_static(m)
        else:
            st.error("❌ Could not determine the best hospital.")
    else:
        st.error("❌ No hospitals found nearby.")
