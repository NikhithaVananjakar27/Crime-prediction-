import streamlit as st
import pandas as pd
import numpy as np
import praw
import folium
from streamlit_folium import st_folium
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from datetime import datetime
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import OrdinalEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

trained_scaler = None
trained_kmeans = None


def feature_engineering(df):
    # --- Object of Search & Weapon Danger ---
    def map_object_of_search(crime):
        if crime in ['HOMICIDE', 'ROBBERY', 'ASSAULT', 'DOMESTIC VIOLENCE','KIDNAPPING','SEXUAL ASSAULT']:
            return 'Offensive weapons'
        elif crime in ['FIREARM OFFENSE']:
            return 'Firearms'
        elif crime in ['BURGLARY', 'VEHICLE - STOLEN', 'SHOPLIFTING', 'FRAUD', 'IDENTITY THEFT','COUNTERFEITING']:
            return 'Stolen goods'
        elif crime in ['DRUG OFFENSE']:
            return 'Controlled drugs'
        else:
            return 'Other'
    df['Object_of_Search'] = df['Crime Description'].apply(map_object_of_search)
    df["Hour"] = df["Date of Occurrence"].dt.hour
    weapon_mapping = {'Other': 0,'Stolen goods' : 1,'Offensive weapons' : 2,'Firearms' : 3,'Controlled drugs' : 4}
    df['Weapon_Danger_Level'] = df['Object_of_Search'].map(weapon_mapping)

    gender_encoder = OrdinalEncoder(categories=[['F','M','X']])
    weapon_enc = OrdinalEncoder(categories=[['Other' ,'Blunt Object','Firearm', 'Knife' ,'Poison' ,'Explosives']],
                                dtype=int,
                                handle_unknown="use_encoded_value",
                                unknown_value=-1)
    df["Weapon_encoded"] = weapon_enc.fit_transform(df[["Weapon Used"]].astype(str))
    df["Genderencode"] = gender_encoder.fit_transform(df[["Victim Gender"]].astype(str))
    df["Closed_Encoded"] = (df["Case Closed"] == "Yes").astype(int)

    # city coords mapping
    city_coords = {
        "Ahmedabad": (23.0225, 72.5714),"Chennai": (13.0827, 80.2707),"Ludhiana": (30.9005, 75.8573),
        "Pune": (18.5204, 73.8567),"Delhi": (28.7041, 77.1025),"Mumbai": (19.0760, 72.8777),
        "Surat": (21.1702, 72.8311),"Visakhapatnam": (17.6868, 83.2185),"Bangalore": (12.9716, 77.5946),
        "Kolkata": (22.5726, 88.3639),"Ghaziabad": (28.6692, 77.4538),"Hyderabad": (17.3850, 78.4867),
        "Jaipur": (26.9124, 75.7873),"Lucknow": (26.8467, 80.9462),"Bhopal": (23.2599, 77.4126),
        "Patna": (25.5941, 85.1376),"Kanpur": (26.4499, 80.3319),"Varanasi": (25.3176, 82.9739),
        "Nagpur": (21.1458, 79.0882),"Meerut": (28.9845, 77.7064),"Thane": (19.2183, 72.9781),
        "Indore": (22.7196, 75.8577),"Rajkot": (22.3039, 70.8022),"Vasai": (19.3919, 72.8397),
        "Agra": (27.1767, 78.0081),"Kalyan": (19.2403, 73.1305),"Nashik": (19.9975, 73.7898),
        "Srinagar": (34.0837, 74.7973),"Faridabad": (28.4089, 77.3178)
    }
    df["Lat"] = df["City"].map(lambda x: city_coords.get(x, (np.nan, np.nan))[0])
    df["Lon"] = df["City"].map(lambda x: city_coords.get(x, (np.nan, np.nan))[1])

    def dangerzone(domain):
        if domain in ['Other Crime', 'Fire Accident']:
            return 0
        elif domain in ['Traffic Fatality', 'Violent Crime']:
            return 1
        else:
            return 0
    df["DangerZone"] = df["Crime Domain"].apply(dangerzone)

    crime_encoder = OrdinalEncoder(categories=[[
        'PUBLIC INTOXICATION','TRAFFIC VIOLATION','VANDALISM','SHOPLIFTING','CYBERCRIME',
        'COUNTERFEITING','IDENTITY THEFT','FRAUD','VEHICLE - STOLEN','ILLEGAL POSSESSION',
        'BURGLARY','EXTORTION','DRUG OFFENSE','ARSON','FIREARM OFFENSE','ROBBERY',
        'DOMESTIC VIOLENCE','ASSAULT','SEXUAL ASSAULT','KIDNAPPING','HOMICIDE']],
        dtype=int,handle_unknown="use_encoded_value",unknown_value=-1)
    df['Crime_Encoded'] = crime_encoder.fit_transform(df[["Crime Description"]].astype(str))

    df['DayorNight'] = df['Hour'].apply(lambda x : 0 if x < 18 else 1)
    df['Month'] = df["Date of Occurrence"].dt.month
    df['Year'] = df["Date of Occurrence"].dt.year
    df['weekday'] = df["Date of Occurrence"].dt.weekday

    def month_to_india_season(month):
        if month in [1, 2]: return 0
        elif 3 <= month <= 5: return 1
        elif 6 <= month <= 9: return 2
        else: return 3
    df['Season'] = df['Month'].apply(month_to_india_season)

    def agebucketingg(age):
        if age >= 65: return 0
        elif 55 <= age <= 64: return 1
        elif 35 <= age <= 54: return 2
        elif 25 <= age <= 34: return 3
        elif 18 <= age <= 24: return 4
        elif 12 <= age <= 17: return 5
        else: return 6
    df['Agebucketing'] = df['Victim Age'].apply(agebucketingg)

    df['Hour_sin'] = np.sin(2*np.pi*df['Hour']/24)
    df['Hour_cos'] = np.cos(2*np.pi*df['Hour']/24)
    df['Month_sin'] = np.sin(2*np.pi*df['Month']/12)
    df['Month_cos'] = np.cos(2*np.pi*df['Month']/12)
    df['weekday_sin'] = np.sin(2*np.pi*df['weekday']/6)
    df['weekday_cos'] = np.cos(2*np.pi*df['weekday']/6)

    coords = df[['Lat', 'Lon']].copy()
    coords_scaled = trained_scaler.transform(coords)   # reuse training scaler
    df['Hotspot_ID'] = trained_kmeans.predict(coords_scaled)
    df['ClusterDist'] = np.min(trained_kmeans.transform(coords_scaled), axis=1)

    # Calculate cluster averages from training data
    # cluster averages from training data
    df['Past_7day_CrimeCount'] = 0.0
    df['Past_30day_CrimeCount'] = 0.0
    cluster_stats = df.groupby('Hotspot_ID')[['Past_7day_CrimeCount', 'Past_30day_CrimeCount']].mean()

    hotspot_id = df['Hotspot_ID'].iloc[0]  # the predicted cluster

    if hotspot_id in cluster_stats.index:
        df['Past_7day_CrimeCount'] = cluster_stats.loc[hotspot_id, 'Past_7day_CrimeCount']
        df['Past_30day_CrimeCount'] = cluster_stats.loc[hotspot_id, 'Past_30day_CrimeCount']
    else:
        # fallback if cluster not seen in training (rare)
        df['Past_7day_CrimeCount'] = df['Past_7day_CrimeCount'].mean()
        df['Past_30day_CrimeCount'] = df['Past_30day_CrimeCount'].mean()

    df = df.reset_index()
    return df


# ------------------ CONFIG ------------------
st.set_page_config(page_title="Crime Prediction Platform", layout="wide")
@st.cache_data
def load_data():
    

    df1 = pd.read_csv(r"C:\Users\KONDAPANAIDU\Downloads\crime_dataset_india.csv")
    df = df1.sample(n=5000,random_state = 41)
    df.dropna(subset=["Weapon Used"], inplace=True)
    for col in ["Date of Occurrence", "Time of Occurrence"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["Hour"] = df["Time of Occurrence"].dt.hour.fillna(0).astype(int)

    # ---- all your feature engineering kept as-is ----
    def map_object_of_search(crime):
        if crime in ['HOMICIDE', 'ROBBERY', 'ASSAULT', 'DOMESTIC VIOLENCE','KIDNAPPING','SEXUAL ASSAULT']:
            return 'Offensive weapons'
        elif crime in ['FIREARM OFFENSE']:
            return 'Firearms'
        elif crime in ['BURGLARY', 'VEHICLE - STOLEN', 'SHOPLIFTING', 'FRAUD', 'IDENTITY THEFT','COUNTERFEITING']:
            return 'Stolen goods'
        elif crime in ['DRUG OFFENSE']:
            return 'Controlled drugs'
        else:
            return 'Other'
    df['Object_of_Search'] = df['Crime Description'].apply(map_object_of_search)

    weapon_mapping = {'Other': 0,'Stolen goods' : 1,'Offensive weapons' : 2,'Firearms' : 3,'Controlled drugs' : 4}
    df['Weapon_Danger_Level'] = df['Object_of_Search'].map(weapon_mapping)

    gender_encoder = OrdinalEncoder(categories=[['F','M','X']])
    weapon_enc = OrdinalEncoder(categories=[['Other' ,'Blunt Object','Firearm', 'Knife' ,'Poison' ,'Explosives']],
                                dtype=int,
                                handle_unknown="use_encoded_value",
                                unknown_value=-1)
    df["Weapon_encoded"] = weapon_enc.fit_transform(df[["Weapon Used"]].astype(str))
    df["Genderencode"] = gender_encoder.fit_transform(df[["Victim Gender"]].astype(str))
    df["Closed_Encoded"] = (df["Case Closed"] == "Yes").astype(int)

    # city coords mapping
    city_coords = {
        "Ahmedabad": (23.0225, 72.5714),"Chennai": (13.0827, 80.2707),"Ludhiana": (30.9005, 75.8573),
        "Pune": (18.5204, 73.8567),"Delhi": (28.7041, 77.1025),"Mumbai": (19.0760, 72.8777),
        "Surat": (21.1702, 72.8311),"Visakhapatnam": (17.6868, 83.2185),"Bangalore": (12.9716, 77.5946),
        "Kolkata": (22.5726, 88.3639),"Ghaziabad": (28.6692, 77.4538),"Hyderabad": (17.3850, 78.4867),
        "Jaipur": (26.9124, 75.7873),"Lucknow": (26.8467, 80.9462),"Bhopal": (23.2599, 77.4126),
        "Patna": (25.5941, 85.1376),"Kanpur": (26.4499, 80.3319),"Varanasi": (25.3176, 82.9739),
        "Nagpur": (21.1458, 79.0882),"Meerut": (28.9845, 77.7064),"Thane": (19.2183, 72.9781),
        "Indore": (22.7196, 75.8577),"Rajkot": (22.3039, 70.8022),"Vasai": (19.3919, 72.8397),
        "Agra": (27.1767, 78.0081),"Kalyan": (19.2403, 73.1305),"Nashik": (19.9975, 73.7898),
        "Srinagar": (34.0837, 74.7973),"Faridabad": (28.4089, 77.3178)
    }
    df["Lat"] = df["City"].map(lambda x: city_coords.get(x, (np.nan, np.nan))[0])
    df["Lon"] = df["City"].map(lambda x: city_coords.get(x, (np.nan, np.nan))[1])

    def dangerzone(domain):
        if domain in ['Other Crime', 'Fire Accident']:
            return 0
        elif domain in ['Traffic Fatality', 'Violent Crime']:
            return 1
        else:
            return 0
    df["DangerZone"] = df["Crime Domain"].apply(dangerzone)

    crime_encoder = OrdinalEncoder(categories=[[
        'PUBLIC INTOXICATION','TRAFFIC VIOLATION','VANDALISM','SHOPLIFTING','CYBERCRIME',
        'COUNTERFEITING','IDENTITY THEFT','FRAUD','VEHICLE - STOLEN','ILLEGAL POSSESSION',
        'BURGLARY','EXTORTION','DRUG OFFENSE','ARSON','FIREARM OFFENSE','ROBBERY',
        'DOMESTIC VIOLENCE','ASSAULT','SEXUAL ASSAULT','KIDNAPPING','HOMICIDE']],
        dtype=int,handle_unknown="use_encoded_value",unknown_value=-1)
    df['Crime_Encoded'] = crime_encoder.fit_transform(df[["Crime Description"]].astype(str))

    df['DayorNight'] = df['Hour'].apply(lambda x : 0 if x < 18 else 1)
    df['Month'] = df["Date of Occurrence"].dt.month
    df['Year'] = df["Date of Occurrence"].dt.year
    df['weekday'] = df["Date of Occurrence"].dt.weekday

    def month_to_india_season(month):
        if month in [1, 2]: return 0
        elif 3 <= month <= 5: return 1
        elif 6 <= month <= 9: return 2
        else: return 3
    df['Season'] = df['Month'].apply(month_to_india_season)

    def agebucketingg(age):
        if age >= 65: return 0
        elif 55 <= age <= 64: return 1
        elif 35 <= age <= 54: return 2
        elif 25 <= age <= 34: return 3
        elif 18 <= age <= 24: return 4
        elif 12 <= age <= 17: return 5
        else: return 6
    df['Agebucketing'] = df['Victim Age'].apply(agebucketingg)

    df['Hour_sin'] = np.sin(2*np.pi*df['Hour']/24)
    df['Hour_cos'] = np.cos(2*np.pi*df['Hour']/24)
    df['Month_sin'] = np.sin(2*np.pi*df['Month']/12)
    df['Month_cos'] = np.cos(2*np.pi*df['Month']/12)
    df['weekday_sin'] = np.sin(2*np.pi*df['weekday']/6)
    df['weekday_cos'] = np.cos(2*np.pi*df['weekday']/6)

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    coords = df[['Lat', 'Lon']].copy()
    scaler = StandardScaler()
    
    coords_scaled = scaler.fit_transform(coords)
    kmeans = KMeans(n_clusters=10, random_state=42)
    df['Hotspot_ID'] = kmeans.fit_predict(coords_scaled)
    df['ClusterDist'] = np.min(kmeans.transform(coords_scaled), axis=1)

    # Save trained scaler & kmeans globally
    global trained_scaler, trained_kmeans
    trained_scaler = scaler
    trained_kmeans = kmeans
    
    

    df['Crime_Happened'] = 1
    df = df.sort_values(['Hotspot_ID', 'Date of Occurrence'])
    df = df.set_index('Date of Occurrence')
    df['Past_7day_CrimeCount'] = df.groupby('Hotspot_ID')['Crime_Happened'].rolling('7D', min_periods=1).sum().reset_index(0, drop=True)
    df['Past_30day_CrimeCount'] = df.groupby('Hotspot_ID')['Crime_Happened'].rolling('30D', min_periods=1).sum().reset_index(0, drop=True)
    df['Past_7day_CrimeCount'] /= df.groupby('Hotspot_ID')['Crime_Happened'].transform('max')
    df['Past_30day_CrimeCount'] /= df.groupby('Hotspot_ID')['Crime_Happened'].transform('max')
    df = df.reset_index()
    return df, scaler, kmeans

df, trained_scaler, trained_kmeans = load_data()


# ------------------ FETCH LIVE TWEETS ------------------

# Reddit API setup
reddit = praw.Reddit(
    client_id="5UynTSP4Rd1KyqlehSaFXg",
    client_secret="81fZG2Gt8urtgWrlJ3DJJxrSQ5vjAA",
    user_agent="python:myredditbot:v1.0 (by /u/luckyprey)"
)

@st.cache_data(ttl=300)
def fetch_reddit_crime_posts(subreddit="india", limit=10):
    try:
        posts_data = []

        for post in reddit.subreddit(subreddit).hot(limit=limit*2):
            if post.stickied:
                continue  # Skip stickied posts

            # Example: Extracting info from post title or body (you may need NLP or regex)
            title = post.title.lower()
            body = post.selftext.lower()

            # Dummy extraction logic (you need to improve this based on your data)
            city = "Unknown"
            time_ = None
            date_ = datetime.fromtimestamp(post.created_utc).date()
            crime_type = "Unknown"
            weapon = "Unknown"

            # Example patterns (you can expand with regex or dictionaries)
            cities = ['Mumbai', 'Kolkata', 'Delhi', 'Ghaziabad', 'Jaipur', 'Chennai',
                        'Rajkot', 'Lucknow', 'Srinagar', 'Hyderabad', 'Bangalore',
                        'Bhopal', 'Patna', 'Kanpur', 'Surat', 'Ahmedabad', 'Ludhiana',
                        'Visakhapatnam', 'Thane', 'Nashik', 'Agra', 'Vasai', 'Varanasi',
                        'Meerut', 'Kalyan', 'Pune', 'Nagpur', 'Faridabad', 'Indore']
            crime_types = ["robbery", "assault", "homicide", "arson", "fraud","drugs"]
            weapons = ["knife", "gun", "fire", "bat"]

            for c in cities:
                if c in title or c in body:
                    city = c.title()
                    break

            for ct in crime_types:
                if ct in title or ct in body:
                    crime_type = ct.title()
                    break

            for w in weapons:
                if w in title or w in body:
                    weapon = w.title()
                    break

            time_ = datetime.fromtimestamp(post.created_utc).strftime("%H:%M:%S")

            posts_data.append({
                "City": city,
                "Date": date_,
                "Time": time_,
                "Crime Type": crime_type,
                "Weapon": weapon,
                "Title": post.title,
                "Body": post.selftext
            })

            if len(posts_data) >= limit:
                break

        return pd.DataFrame(posts_data)

    except Exception as e:
        st.error(f"Error fetching Reddit posts: {e}")
        return pd.DataFrame()

# ------------------ MODEL TRAINING ------------------
X = df[['Past_30day_CrimeCount','ClusterDist','Hour','Month_sin','Month_cos','Year','weekday_sin','weekday_cos','Season','DayorNight','Agebucketing','Genderencode','Past_7day_CrimeCount','Closed_Encoded','Weapon_Danger_Level']]
y = df["DangerZone"] 

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X,y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res,stratify=y_res, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=2000,max_depth=15,
    min_samples_split=10,min_samples_leaf=5,
    max_features='sqrt',bootstrap=True,
    random_state=42,class_weight='balanced'
)
model.fit(X_res,y_res)

# ------------------ STREAMLIT UI ------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📑 Data Explorer","🔮 Predict Crime Risk","📍 Insights & Map","🐦 Live Tweets"
])

# ---- Tab 1 ----
with tab1:
    st.header("Explore Dataset")
    st.dataframe(df.head(20))

# ---- Tab 2 ----
with tab2:
    st.header("Predict Crime Risk & Danger Zone")
    col1, col2 = st.columns(2)

    with col1:
        city_input = st.selectbox("Select City", df["City"].dropna().unique())
        crime_input = st.selectbox("Select Crime Type", df["Crime Description"].unique())
        domain_input = st.selectbox("Crime Domain", df["Crime Domain"].unique())
        age_input = st.slider("Victim Age", 10, 90, 30)
        closed_input = st.radio("Case Closed?", ["Yes", "No"])
        gender_input = st.selectbox("Victim Gender", ["M","F","X"])
        date_input = st.date_input("Date of Occurrence")
        time_input = st.time_input("Time of Occurrence")
        user_text = st.text_area("Enter related tweet/post (for sentiment analysis)")
    

    with col2:
        sentiment_score = TextBlob(user_text).sentiment.polarity if user_text else 0.0
        st.metric("Sentiment Polarity", round(sentiment_score, 2))

        # Build single-row dataframe
        user_df = pd.DataFrame([{
            "City": city_input,
            "Crime Description": crime_input,
            "Crime Domain": domain_input,
            "Victim Age": age_input,
            "Victim Gender": gender_input,
            "Case Closed": closed_input,
            "Weapon Used": "Other",  # or add user input
            "Date of Occurrence": pd.to_datetime(f"{date_input} {time_input}")
        }])

        # Run feature engineering
        user_features = feature_engineering(user_df)

        # Select model features
        X_pred = user_features[[
            'Past_30day_CrimeCount','ClusterDist','Hour','Month_sin','Month_cos','Year',
            'weekday_sin','weekday_cos','Season','DayorNight','Agebucketing',
            'Genderencode','Past_7day_CrimeCount','Closed_Encoded','Weapon_Danger_Level'
        ]]

        # Predict
        pred = int(model.predict(X_pred)[0])

        if pred == 1 or sentiment_score < -0.3:
            st.error("🚨 Danger Zone! High risk of crime detected.")
        else:
            st.success("✅ Low risk zone.")


# ---- Tab 3 ----
with tab3:
    st.header("Map: Danger Zones & Your Location")
    st.subheader("Visualize cities with high-risk (danger zones)")
    lat_cent = df["Lat"].mean() if not np.isnan(df["Lat"]).all() else 22.0
    lon_cent = df["Lon"].mean() if not np.isnan(df["Lon"]).all() else 78.0
    m = folium.Map(location=[lat_cent, lon_cent], zoom_start=5)

    # Plot danger zones
    for _, row in df[df["DangerZone"] == 1].iterrows():
        folium.CircleMarker(location=(row["Lat"], row["Lon"]),
                            radius=3,
                            color='red',
                            fill=True,
                            fill_color = 'red',
                            fill_opacity=0.7,
                            popup=f"{row['City']}: {row['Crime Description']}"
                            ).add_to(m)
    # Plot safe zones
    for _, row in df[df["DangerZone"] == 0].iterrows():
        folium.CircleMarker(location=(row["Lat"], row["Lon"]),
                            radius=3,
                            color='green',
                            fill=True,
                            fill_color = 'green',
                            fill_opacity=0.5,
                            popup=f"{row['City']}: {row['Crime Description']}"
                            ).add_to(m)
    st_folium(m, width=900)

    st.subheader("Sample Insights")
    st.write("Most common violent crimes:", df[df["DangerZone"] == 1]["Crime Description"].value_counts().head())
    st.write("Cities with most danger zones:", df[df["DangerZone"] == 1]["City"].value_counts().head())

# ---- Tab 4 ----
with tab4:
    st.header("Reddit Crime Posts (PRAW API)")
    reddit_posts = pd.DataFrame()
    
    subreddit_input = st.text_input("Enter subreddit:", value="india")
    post_limit = st.slider("Number of posts:", 5, 50, 10)
    
    if st.button("Fetch Reddit Posts"):
        with st.spinner("Fetching Reddit posts..."):
            reddit_posts = fetch_reddit_crime_posts(subreddit=subreddit_input, limit=post_limit)
            
            if not reddit_posts.empty:
                st.success(f"Fetched {len(reddit_posts)} posts!")
                st.dataframe(reddit_posts)
                st.write("Last Updated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            else:
                st.warning("No posts found.")
