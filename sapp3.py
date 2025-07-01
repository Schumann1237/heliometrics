# sapp.py
import streamlit as st
from datetime import date
from datetime import date, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from solarWeather import get_weather_data_dynamic
from solarAirQuality import get_air_quality_data_dynamic
from solarRadiation import get_radiation_data_dynamic
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt


def reverse_geocode(lat, lon):
    url = "https://api.bigdatacloud.net/data/reverse-geocode-client"
    params = {
        "latitude": lat,
        "longitude": lon,
        "localityLanguage": "en"
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        city = data.get('city') or data.get('locality') or data.get('principalSubdivision') or 'unknown city'
        country = data.get('countryName', 'unknown country')
        return city, country
    except:
        return "unknown city", "unknown country"

st.set_page_config(page_title="Solar Weather & Air Quality Dashboard", layout="wide")

# Sidebar for page navigation
st.sidebar.title("Navigation")

st.sidebar.markdown("---")

# Initialize session state for page selection
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Data View"

# Navigation buttons  side page
pages = {
    "Data View": "Data View",
    "Visualization": "Visualization",
    "Predictions": "Predictions",
    "Estimation": "Estimation",
    "About Us": "About Us"
}

# 渲染侧边导航按钮
for label, page in pages.items():
    if st.sidebar.button(label, use_container_width=True, 
                         type="primary" if st.session_state.current_page == page else "secondary"):
        st.session_state.current_page = page


page = st.session_state.current_page


# Top section - User inputs (always visible)
st.title("Solar Weather & Air Quality Dashboard")
st.markdown("Go ahead and enter the location and date you're interested in and you'll be amazed at the detailed historical weather and air quality data you can access!")

# Initialize session state for coordinates
if 'latitude' not in st.session_state:
    st.session_state.latitude = 4.2105
if 'longitude' not in st.session_state:
    st.session_state.longitude = 101.9758

# User input section - choice between sliders and map
st.subheader("Select Location")
input_method = st.radio(
    "Choose how to select coordinates:",
    [ "Click on Map","User Input"],
    horizontal=True
)


if input_method == "User Input":
    # User input section with sliders
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.latitude = st.number_input("Latitude", value=52.52, format="%.4f")
        st.info("Select latitude between -90° (South Pole) and 90° (North Pole)")
        
    with col2:
        st.session_state.longitude = st.number_input("Longitude", value=13.41, format="%.4f")
        st.info("Select longitude between -180° (West) and 180° (East)")

# 初始化 folium 地图
m = folium.Map(
    location=[st.session_state.latitude, st.session_state.longitude],
    zoom_start=8,  
)

# 加 marker
folium.Marker(
    [st.session_state.latitude, st.session_state.longitude],
    popup=f"{st.session_state.latitude:.4f}°N, {st.session_state.longitude:.4f}°E",
    tooltip="Selected Location - Click map to change location",
    icon=folium.Icon(color='red', icon='info-sign')
).add_to(m)

# 添加点击坐标提示
m.add_child(folium.LatLngPopup())

# 🌟 只调用一次 st_folium
map_data = st_folium(m, width=725, height=500, returned_objects=["last_clicked", "last_object_clicked"])

# 如果是点击选点模式 & 用户点击了地图
if input_method == "Click on Map":
    st.write("**Click anywhere on the map to select coordinates**")
    
    if map_data and map_data["last_clicked"]:
        st.session_state.latitude = map_data["last_clicked"]["lat"]
        st.session_state.longitude = map_data["last_clicked"]["lng"]
        st.rerun()


# Use session state values for the rest of the application
latitude = st.session_state.latitude
longitude = st.session_state.longitude

# 3. 调用函数获取城市国家名
try:
    with st.spinner("Retrieving city and country name..."):
        city, country = reverse_geocode(latitude, longitude)
except Exception as e:
    city, country = "unknown city", "unknown country"
    st.warning("cant obtain the city and country name, please check your internet connection")



st.markdown(f"### Your selected location is: **{city}**,  **{country}**!")
# Display map (always visible)
st.info(f"Selected coordinates: {st.session_state.latitude:.4f}°N, {st.session_state.longitude:.4f}°E")

# Date input section    
col3, col4 = st.columns(2)  

today = date.today()
with col3:
    start_date = st.date_input("Start Date", value=today.replace(month=1, day=1), max_value=today)
with col4:
    end_date = st.date_input("End Date", value=today, max_value=today)

if start_date > end_date:
    st.error("Start date must be before end date!")
else:
    st.success(f"Selected range: {start_date} to {end_date}")


# Initialize session state for data storage
if 'combined_df' not in st.session_state:
    st.session_state.combined_df = pd.DataFrame()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Query button
if st.button("Search"):
    with st.spinner("Retrieving weather and air quality data... Please wait~"):
    #   call dataframe 
        weather_df = get_weather_data_dynamic(latitude, longitude, start_date, end_date)
        air_quality_df = get_air_quality_data_dynamic(latitude, longitude, start_date, end_date)
        radiation_df = get_radiation_data_dynamic(latitude, longitude, start_date, end_date)
       
        if weather_df.empty or air_quality_df.empty or radiation_df.empty:
            st.error("Unable to retrieve data. Please check your input or try again later.")
            st.session_state.data_loaded = False
        else:
            # 先合并前两个DataFrame
            temp_df = pd.merge(weather_df, air_quality_df, on="date", how="inner")
            # 再合并第三个DataFrame
            st.session_state.combined_df = pd.merge(temp_df, radiation_df, on="date", how="inner")
            st.session_state.data_loaded = True
            st.success("Data retrieval successful!")

# Divider
st.divider()

# Page content based on selection
if page == "Data View":
    st.header("Data Table View")
    
    if st.session_state.data_loaded and not st.session_state.combined_df.empty:
        # Data statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(st.session_state.combined_df))
        with col2:
            st.metric("Avg Temperature", f"{st.session_state.combined_df['temperature_2m'].mean():.1f}°C")
        with col3:
            st.metric("Total Rainfall", f"{st.session_state.combined_df['rain'].sum():.1f}mm")
        with col4:
            st.metric("Avg Wind Speed", f"{st.session_state.combined_df['wind_speed_10m'].mean():.1f}m/s")
        
        # Display dataframe
        st.dataframe(
            st.session_state.combined_df,
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = st.session_state.combined_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"weather_air_quality_{start_date}_{end_date}.csv",
            mime="text/csv"
        )
    else:
        st.info("Please query the data first to view the data table.")


    st.write("Exiting variables: ", st.session_state.combined_df.columns.tolist())
 
    

elif page == "Visualization":
    st.header("Data Visualization")
    
    if st.session_state.data_loaded and not st.session_state.combined_df.empty:
        df = st.session_state.combined_df.copy()
        
        # Convert date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Visualization tabs
        tab1, tab2, tab3 = st.tabs(["Weather Data", "Air Quality", "Association Analysis"])
        
        with tab1:
            st.subheader("Weather Parameters")
            
            # Temperature and Rain
            fig1 = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Temperature (°C)', 'Rainfall (mm)', 'Cloud Cover (%)', 'Wind Speed (m/s)'),
                vertical_spacing=0.09              
            )
            
            # fig1.add_trace(go.Scatter(x=df['date'], y=df['temperature_2m'], 
            #                         name='Temperature', line=dict(color='red')), row=1, col=1)
            fig1.add_trace(go.Box(y=df['temperature_2m'], 
                     name='Temperature', 
                     marker_color='red'), 
              row=1, col=1)
            fig1.add_trace(go.Scatter(x=df['date'], y=df['rain'], 
                                    name='Rain', line=dict(color='blue')), row=1, col=2)
            fig1.add_trace(go.Scatter(x=df['date'], y=df['cloud_cover'], 
                                    name='Cloud Cover', line=dict(color='gray')), row=2, col=1)
            fig1.add_trace(go.Scatter(x=df['date'], y=df['wind_speed_10m'], 
                                    name='Wind Speed', line=dict(color='green')), row=2, col=2)
            
            
            fig1.update_layout(height=900, showlegend=False, title_text="Weather Parameters Over Time")
            st.plotly_chart(fig1, use_container_width=True)
            
            # shortwave_radiation
            fig3 = px.line(df, x='date', y='shortwave_radiation', 
                          title='Shortwave Radiation Over Time',
                          color_discrete_sequence=['darkblue'])
            st.plotly_chart(fig3, use_container_width=True)
        
        with tab2:
            st.subheader("Air Quality Parameters")
            
            # Air Quality metrics
            fig2 = make_subplots(
                rows=2, cols=2,
                subplot_titles=('PM10 (μg/m³)', 'PM2.5 (μg/m³)', 'UV Index', 'Dust (μg/m³)'),
                vertical_spacing=0.09
            )
            
            fig2.add_trace(go.Scatter(x=df['date'], y=df['pm10'], 
                                    name='PM10', line=dict(color='orange')), row=1, col=1)
            fig2.add_trace(go.Scatter(x=df['date'], y=df['pm2_5'], 
                                    name='PM2.5', line=dict(color='red')), row=1, col=2)
            fig2.add_trace(go.Scatter(x=df['date'], y=df['uv_index'], 
                                    name='UV Index', line=dict(color='purple')), row=2, col=1)
            fig2.add_trace(go.Scatter(x=df['date'], y=df['dust'], 
                                    name='Dust', line=dict(color='brown')), row=2, col=2)
            
            fig2.update_layout(height=900, showlegend=False, title_text="Air Quality Parameters Over Time")
            st.plotly_chart(fig2, use_container_width=True)
            
            # Aerosol Optical Depth
            fig3 = px.line(df, x='date', y='aerosol_optical_depth', 
                          title='Aerosol Optical Depth Over Time',
                          color_discrete_sequence=['darkblue'])
            st.plotly_chart(fig3, use_container_width=True)
        
        
        with tab3:
            st.subheader("Association Analysis")
            
            # Correlation heatmap
            # 检查并处理数据
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

            # 如果没有自动检测到数值列，尝试强制转换
            if len(numeric_cols) < 2:
                df_numeric = df.apply(pd.to_numeric, errors='coerce')
                numeric_cols = df_numeric.dropna(axis=1, how='all').columns.tolist()

            # 过滤掉常数列
            numeric_cols = [col for col in numeric_cols if df[col].nunique() > 1]

            # 绘制热力图
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                fig4 = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Parameter Correlation Matrix",
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.error("Can't draw heatmap, no numeric columns")
            
            # Daily averages
            df_daily = df.set_index('date').resample('D').mean()
            
            col1, col2 = st.columns(2)
            with col1:
                fig5 = px.scatter(df_daily, x='temperature_2m', y='pm2_5',
                                title='Temperature vs PM2.5 (Daily Averages)',
                                trendline='ols')
                st.plotly_chart(fig5, use_container_width=True)
            
            with col2:
                fig6 = px.scatter(df_daily, x='cloud_cover', y='uv_index',
                                title='Cloud Cover vs UV Index (Daily Averages)',
                                trendline='ols')
                st.plotly_chart(fig6, use_container_width=True)
                            
    else:
        st.info("Please query the data first to view the visualisation chart.")

elif page == "Predictions":
    st.header("Features Predictions")
    
    if not st.session_state.data_loaded or st.session_state.combined_df.empty:
        st.info("🔍 Please query the data first to make predictions.")
    else:
        # Initialize forecast results in session state if not exists
        if 'forecast_results' not in st.session_state:
            st.session_state.forecast_results = {}
        if 'model_performance' not in st.session_state:
            st.session_state.model_performance = {}
        if 'forecast_config' not in st.session_state:
            st.session_state.forecast_config = {}
        
        # 用户选择时间范围和频率
        st.subheader("⚙️ Forecast Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            option = st.radio("🔍 Choose predictive model time range:", ["Days", "Weeks", "Years"])
                              
        
        today = date.today() - timedelta(days=1)

        if option == "Years":
            years = st.number_input("📅 Number of Years", min_value=1, max_value=10, value=1)
        elif option == "Weeks":
            weeks = st.number_input("📅 Number of Weeks", min_value=1, max_value=52, value=4)
        else:  # Days
            days = st.number_input("📅 Number of Days", min_value=1, max_value=365, value=3)

        # 先计算基础单位
        base_units = {
            "Years": 365*24,
            "Weeks": 7*24,
            "Days": 1*24
        }[option]

        # 再根据频率调整
        frequency = "Hourly"
        forecast_periods = base_units * (years if option=="Years" else weeks if option=="Weeks" else days)
        frequency == "Hourly"
        forecast_end_date = today + timedelta(days=forecast_periods/24)
        freq_str = "H"


        # 配置信息展示卡片
        st.info(f"🗓️ **Forecast Configuration**\n"
                f"📅 Forecast Period: {today} → {forecast_end_date}\n"
                f"⏱️ Duration: {forecast_periods} {frequency.lower()} periods\n"
                f"🔄 Frequency: {frequency}\n"
                f"📊 Training Data: {start_date} to {end_date}")

        # 调用你的 forecast 函数
        col1, col2 = st.columns([3, 1])
        with col1:
            run_forecast = st.button("🚀 Run Forecast", type="primary", use_container_width=True)
        with col2:
            if st.session_state.forecast_results:
                clear_forecast = st.button("🗑️ Clear Results", type="secondary", use_container_width=True)
                if clear_forecast:
                    st.session_state.forecast_results = {}
                    st.session_state.model_performance = {}
                    st.session_state.forecast_config = {}
                    st.rerun()

        if run_forecast:
            with st.spinner("📡 Forecasting in progress... This may take a few moments..."):
                try:
                    # Store forecast configuration
                    st.session_state.forecast_config = {
                    'option': option,
                    'frequency': frequency,
                    'periods': forecast_periods,
                    'forecast_start_date': today,           # ✅ Forecast starts today
                    'forecast_end_date': forecast_end_date, # ✅ Calculated forecast end
                    'data_end_date': end_date,              # ✅ User's data collection end date
                    'freq_str': freq_str
                }
                    
                    # Prepare data for forecasting
                    df = st.session_state.combined_df.copy()
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    
                    # Get numeric columns for forecasting (exclude date)
                    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    # Store forecasted results
                    forecast_results = {}
                    model_performance = {}
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, column in enumerate(numeric_columns):
                        status_text.text(f"Forecasting {column}... ({i+1}/{len(numeric_columns)})")
                        
                        # Prepare data for this column
                        data = df[['date', column]].dropna()
                        
                        if len(data) < 10:  # Need minimum data points
                            continue
                            
                        # Create features for time series (enhanced for hourly)
                        data = data.sort_values('date').reset_index(drop=True)
                        data['day_of_year'] = data['date'].dt.dayofyear
                        data['month'] = data['date'].dt.month
                        data['day_of_week'] = data['date'].dt.dayofweek
                        data['hour'] = data['date'].dt.hour
                        data['is_weekend'] = (data['date'].dt.dayofweek >= 5).astype(int)
                        data['days_since_start'] = (data['date'] - data['date'].min()).dt.total_seconds() / (24*3600)
                        
                        if frequency == "Hourly":
                            data['hours_since_start'] = (data['date'] - data['date'].min()).dt.total_seconds() / 3600
                            # Add hourly cyclical features
                            data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
                            data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
                            # Add daily cyclical features
                            data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
                            data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
                        
                        # Create lagged features (adjust for frequency)
                        if frequency == "Daily":
                            lag_periods = [1, 7, 30]  # 1 day, 1 week, 1 month
                            rolling_windows = [7, 30]  # 1 week, 1 month
                        else:  # Hourly
                            lag_periods = [1, 24, 168]  # 1 hour, 1 day, 1 week
                            rolling_windows = [24, 168]  # 1 day, 1 week
                        
                        for lag in lag_periods:
                            if len(data) > lag:
                                data[f'{column}_lag_{lag}'] = data[column].shift(lag)
                        
                        # Rolling statistics
                        for window in rolling_windows:
                            data[f'{column}_rolling_{window}'] = data[column].rolling(window=min(window, len(data))).mean()
                            data[f'{column}_rolling_std_{window}'] = data[column].rolling(window=min(window, len(data))).std()
                        
                        # Drop rows with NaN values
                        data_clean = data.dropna()
                        
                        if len(data_clean) < 5:
                            continue
                        
                        # Prepare features and target
                        feature_cols = [col for col in data_clean.columns if col not in ['date', column]]
                        X = data_clean[feature_cols]
                        y = data_clean[column]
                        
                        # Split data for training and testing
                        if len(data_clean) > 10:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42, shuffle=False
                            )
                        else:
                            X_train, X_test, y_train, y_test = X, X, y, y
                        
                        # Train Random Forest model
                        model = RandomForestRegressor(
                            n_estimators=100,
                            random_state=42,
                            max_depth=10,
                            min_samples_split=2,
                            min_samples_leaf=1
                        )
                        model.fit(X_train, y_train)
                        
                        # Calculate model performance
                        if len(X_test) > 0:
                            y_pred_test = model.predict(X_test)
                            mae = mean_absolute_error(y_test, y_pred_test)
                            mse = mean_squared_error(y_test, y_pred_test)
                            rmse = np.sqrt(mse)
                            model_performance[column] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse}
                        
                        # Generate future dates
                        last_date = data['date'].max()
                        if frequency == "Hourly":
                            start_date = last_date + pd.Timedelta(hours=1)
                        else:
                            start_date = last_date + pd.Timedelta(days=1)

                        future_dates = pd.date_range(
                            start=start_date,
                            periods=forecast_periods,
                            freq=freq_str
                        )

                        
                        # Create future features
                        future_data = pd.DataFrame({'date': future_dates})
                        future_data['day_of_year'] = future_data['date'].dt.dayofyear
                        future_data['month'] = future_data['date'].dt.month
                        future_data['day_of_week'] = future_data['date'].dt.dayofweek
                        future_data['hour'] = future_data['date'].dt.hour
                        future_data['is_weekend'] = (future_data['date'].dt.dayofweek >= 5).astype(int)
                        future_data['days_since_start'] = (future_data['date'] - data['date'].min()).dt.total_seconds() / (24*3600)
                        
                        if frequency == "Hourly":
                            future_data['hours_since_start'] = (future_data['date'] - data['date'].min()).dt.total_seconds() / 3600
                            # Add hourly cyclical features
                            future_data['hour_sin'] = np.sin(2 * np.pi * future_data['hour'] / 24)
                            future_data['hour_cos'] = np.cos(2 * np.pi * future_data['hour'] / 24)
                            # Add daily cyclical features
                            future_data['day_sin'] = np.sin(2 * np.pi * future_data['day_of_week'] / 7)
                            future_data['day_cos'] = np.cos(2 * np.pi * future_data['day_of_week'] / 7)
                        
                        # For lagged features, use recent values
                        recent_values = data[column].tail(max(lag_periods) * 2 if lag_periods else 60).values
                        
                        predictions = []
                        for j in range(forecast_periods):
                            row_features = future_data.iloc[j:j+1].copy()
                            
                            # Add lagged features dynamically
                            for lag in lag_periods:
                                if len(recent_values) >= lag:
                                    if j == 0:
                                        row_features[f'{column}_lag_{lag}'] = recent_values[-lag] if lag <= len(recent_values) else recent_values[-1]
                                    elif j < lag:
                                        row_features[f'{column}_lag_{lag}'] = recent_values[-(lag-j)] if lag-j <= len(recent_values) else recent_values[-1]
                                    else:
                                        row_features[f'{column}_lag_{lag}'] = predictions[j-lag]
                                else:
                                    row_features[f'{column}_lag_{lag}'] = recent_values[-1] if len(recent_values) > 0 else 0
                            
                            # Add rolling features
                            for window in rolling_windows:
                                if j < window:
                                    # Use recent historical values
                                    if len(recent_values) >= window:
                                        row_features[f'{column}_rolling_{window}'] = recent_values[-window:].mean()
                                        row_features[f'{column}_rolling_std_{window}'] = recent_values[-window:].std()
                                    else:
                                        row_features[f'{column}_rolling_{window}'] = recent_values.mean() if len(recent_values) > 0 else 0
                                        row_features[f'{column}_rolling_std_{window}'] = recent_values.std() if len(recent_values) > 1 else 0
                                else:
                                    # Use recent predictions
                                    recent_preds = predictions[j-window:j]
                                    row_features[f'{column}_rolling_{window}'] = np.mean(recent_preds)
                                    row_features[f'{column}_rolling_std_{window}'] = np.std(recent_preds) if len(recent_preds) > 1 else 0
                            
                            # Fill any missing columns with 0
                            for col in feature_cols:
                                if col not in row_features.columns:
                                    row_features[col] = 0
                            
                            # Ensure columns are in the same order as training
                            row_features = row_features[feature_cols]
                            
                            # Make prediction
                            pred = model.predict(row_features)[0]
                            predictions.append(pred)
                        
                        # Store results
                        forecast_results[column] = {
                            'dates': future_dates,
                            'predictions': predictions,
                            'historical_data': data[['date', column]]
                        }
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(numeric_columns))
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    if forecast_results:
                        st.success(f"✅ Successfully forecasted {len(forecast_results)} variables!")
                        
                        # Store results in session state
                        st.session_state.forecast_results = forecast_results
                        st.session_state.model_performance = model_performance
                        
                except Exception as e:
                    st.error(f"❌ Error during forecasting: {str(e)}")
                    st.write("Please check your data and try again.")

        # Display results if they exist (persistent display)
        if st.session_state.forecast_results:
            st.divider()
            st.subheader("📊 Forecast Results")
            
            # Get config for display
            config = st.session_state.forecast_config
            
            # Display forecast summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📈 Variables Forecasted", len(st.session_state.forecast_results))
            with col2:
                st.metric("📅 Forecast Period", f"{config.get('periods', 'N/A')} {config.get('frequency', '').lower()}")
            with col3:
                # Show forecast start date (today)
                forecast_start = config.get('forecast_start_date', 'N/A')
                if isinstance(forecast_start, date):
                    forecast_start = str(forecast_start)
                st.metric("🎯 Forecast Start", forecast_start)
            with col4:
                # Show forecast end date (calculated)
                forecast_end = config.get('forecast_end_date', 'N/A')
                if isinstance(forecast_end, date):
                    forecast_end = str(forecast_end)
                st.metric("🏁 Forecast End", forecast_end)
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["Forecast Data", "Interactive Charts", "Combined View", "Performance"])
            
            with tab1:
                st.subheader("Forecasted Data Table")
                
                # Create combined forecast dataframe
                if st.session_state.forecast_results:
                    future_dates = list(st.session_state.forecast_results.values())[0]['dates']
                    forecast_df = pd.DataFrame({'date': future_dates})
                    
                            # 添加格式化（关键修改）
                    forecast_df['date'] = forecast_df['date'].dt.strftime('%Y-%m-%d')   
                    
                    for column, results in st.session_state.forecast_results.items():
                        forecast_df[f'{column}_forecast'] = results['predictions']
                    
                    # Display forecast dataframe with better formatting
                    st.dataframe(
                        forecast_df.style.format({
                            col: '{:.2f}' for col in forecast_df.columns if col != 'date'
                        }),
                        use_container_width=True,
                        height=400
                    )
                        # Store forecast_df in session state
                    st.session_state.forecast_df = forecast_df
                    
                    # Download button for forecast data
                    csv_forecast = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="Download Forecast CSV",
                        data=csv_forecast,
                        file_name=f"forecast_{config.get('frequency', 'unknown').lower()}_{config.get('start_date', '')}_{str(config.get('end_date', ''))[:10]}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            

            
            with tab2:
                st.subheader("Interactive Forecast Charts")
                
                # Variable selector for individual charts
                selected_vars = st.multiselect(
                    "Select variables to visualize:",
                    options=list(st.session_state.forecast_results.keys()),
                    default=list(st.session_state.forecast_results.keys())[:2]  # Show first 2 by default       
                )
                
                for column in selected_vars:
                    if column in st.session_state.forecast_results:
                        results = st.session_state.forecast_results[column]
                        
                        # Create matplotlib figure
                        fig, ax = plt.subplots(figsize=(20, 6))
                        
                        # Historical data
                        historical = results['historical_data']
                        ax.plot(
                            historical['date'],
                            historical[column],
                            color='blue',
                            linewidth=2,
                            label=f'Historical {column}'
                        )
                        
                        # Forecast data
                        ax.plot(
                            results['dates'],
                            results['predictions'],
                            color='red',
                            linewidth=2,
                            linestyle='--',
                            label=f'Forecast {column}'
                        )
                        
                        # Add vertical line to separate historical and forecast
                        last_historical_date = historical['date'].max()
                        ax.axvline(
                            x=last_historical_date,
                            color='green',
                            linestyle='--',
                            alpha=0.7,
                            label='Forecast Start'
                        )
                        
                        # Formatting
                        ax.set_title(f'Historical vs Forecast: {column}', fontsize=14, fontweight='bold')
                        ax.set_xlabel('Date', fontsize=12)
                        ax.set_ylabel(f'{column}', fontsize=12)
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        # Rotate x-axis labels for better readability
                        plt.xticks(rotation=45)
                        
                        # Adjust layout to prevent label cutoff
                        plt.tight_layout()
                        
                        # Display in Streamlit
                        st.pyplot(fig)
                        
                        # Close the figure to free memory
                        plt.close(fig)
   
            with tab4:
                st.subheader("Model Performance Metrics")
                
                if st.session_state.model_performance:
                    # Create performance dataframe
                    perf_data = []
                    for var, metrics in st.session_state.model_performance.items():
                        perf_data.append({
                            'Variable': var,
                            'MAE': f"{metrics['MAE']:.4f}",
                            'MSE': f"{metrics['MSE']:.4f}",
                            'RMSE': f"{metrics['RMSE']:.4f}"
                        })
                    
                    perf_df = pd.DataFrame(perf_data)
                    
                    # Display performance table
                    st.dataframe(
                        perf_df.style.set_properties(**{
                            'background-color': 'lightblue',
                            'color': 'black',
                            'border-color': 'white'
                        }),
                        use_container_width=True
                    )
                    
                    # Visualize performance metrics
                    fig_perf = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('Mean Absolute Error (MAE)', 'Root Mean Square Error (RMSE)'),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                    )
                    
                    variables = list(st.session_state.model_performance.keys())
                    mae_values = [st.session_state.model_performance[var]['MAE'] for var in variables]
                    rmse_values = [st.session_state.model_performance[var]['RMSE'] for var in variables]
                    
                    fig_perf.add_trace(
                        go.Bar(name='MAE', x=variables, y=mae_values, marker_color='lightblue'),
                        row=1, col=1
                    )
                    
                    fig_perf.add_trace(
                        go.Bar(name='RMSE', x=variables, y=rmse_values, marker_color='lightcoral'),
                        row=1, col=2
                    )
                    
                    fig_perf.update_layout(
                        title_text='Model Performance Metrics Comparison',
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_perf, use_container_width=True)
                    
                    # Performance insights
                    best_mae_var = min(st.session_state.model_performance.items(), key=lambda x: x[1]['MAE'])
                    worst_mae_var = max(st.session_state.model_performance.items(), key=lambda x: x[1]['MAE'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"**Best Performance (Lowest MAE)**\n\n"
                                 f"Variable: {best_mae_var[0]}\n\n"
                                 f"MAE: {best_mae_var[1]['MAE']:.4f}")
                    with col2:
                        st.warning(f"**Needs Improvement (Highest MAE)**\n\n"
                                 f"Variable: {worst_mae_var[0]}\n\n"
                                 f"MAE: {worst_mae_var[1]['MAE']:.4f}")
                    
                    st.info("**Model Performance Guide:**\n"
                           "- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values\n"
                           "- **RMSE (Root Mean Square Error)**: Square root of average squared differences\n"
                           "- **Lower values indicate better model performance**\n"
                           "- Consider the scale of your data when interpreting these metrics")
                
                else:
                    st.warning("No performance metrics available. This may happen with very small datasets.")
        
        else:
            st.info("Run a forecast to see results here. Your forecast data will be displayed persistently like the data view.")



elif page == "Estimation":
    st.header("Solar Performance Estimator")

    if st.session_state.data_loaded:
   
        daily_sun_hours = 12
        avg_radiation = 0

        if 'forecast_df' in st.session_state and not st.session_state.forecast_df.empty:
            df = st.session_state.forecast_df.copy()
            st.info("Using forecasted data for estimation")
            avg_radiation = df['shortwave_radiation_forecast'].mean()

            df2 = st.session_state.combined_df.copy()
            df2['sun_hours'] = df2['shortwave_radiation'].apply(lambda x: 1 if x > 100 else 0)
            daily_sun_hours = df2.groupby(df2['date'].dt.date)['sun_hours'].sum().mean()
        else:
            df = st.session_state.combined_df.copy()
            st.warning("Using historical data (no forecast available)")
            avg_radiation = df['shortwave_radiation'].mean() 
            df['sun_hours'] = df['shortwave_radiation'].apply(lambda x: 1 if x > 100 else 0)
            daily_sun_hours = df.groupby(df['date'].dt.date)['sun_hours'].sum().mean()

        st.subheader("System Configuration")
        col1, col2 = st.columns(2)

        with col1:
            system_size = st.number_input("System Size (m2)", min_value=5.0, value=5.0, step=0.5)
            elec_charge = st.number_input("Electricity Charge (RM/kWh)", min_value=0.1,max_value=1.0, value=0.5, step=0.01)
            cost_per_kw = st.number_input("Cost per kW (RM)", min_value=0.0, max_value= elec_charge, value=0.2,step=0.5)
            # elec_rate = st.number_input("Electricity Rate (RM/kWh)", min_value=0.1, value=0.218, step=0.01)

            years = st.number_input("How many years?", min_value=1, max_value=100, value=20, step=1)
            additional_cost= 5000
            fixed_cost=500
   

        with col2:
            st.metric("Effective Sun Hours/Day", f"{daily_sun_hours:.1f} hours")
            st.metric("Solar Radiation per hour (prediction)", f"{avg_radiation:.0f} Wh/m²")

      
        
        year_list = list(range(0, years))
        revenue_list = []
        cost_list = []
        profit_list = []
        
        
        
                
#         efficiency = 0.98 **years
#         maintenance_cost = 40 ** (1/efficiency)
#         energy_generated_per_day= system_size *  avg_radiation * daily_sun_hours *efficiency
#         total_cost = (energy_generated_per_day * cost_per_kw)* (year*365) + maintenance_cost
#         total_revenue = energy_generated_per_day * elec_charge * (year*365)
#         profit = total_revenue - total_cost

        for years in year_list:
            efficiency = 0.97 ** years  
            maintenance_cost = 100 ** (1 / efficiency)
            panel_unit = system_size / 5

            energy_generated_per_day = system_size * (avg_radiation/1000) * daily_sun_hours * efficiency

            total_cost =  ((energy_generated_per_day * cost_per_kw)/panel_unit)* (years*365) + (maintenance_cost+fixed_cost+additional_cost)*panel_unit
            total_revenue = energy_generated_per_day * elec_charge * (years*365) 
            net_profit = total_revenue - total_cost 

            revenue_list.append(total_revenue)
            cost_list.append(total_cost)
            profit_list.append(net_profit)

        # 生成 Plotly 图表
        import plotly.graph_objects as go
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=year_list,
            y=revenue_list,
            name="Total Revenue",
            line=dict(color='green', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=year_list,
            y=cost_list,
            name="Total Cost",
            line=dict(color='red', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=year_list,
            y=profit_list,
            name="Net Profit",
            line=dict(color='blue', width=3, dash='dot')
        ))

        fig.update_layout(
            title="Solar System Break-Even Analysis",
            xaxis_title="Years",
            yaxis_title="Amount (RM)",
            height=450,
            template="simple_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)


if page == "About Us":
    

    st.title("**Meet our team:**")
    st.write("""
    **Isaac Yang Hao Tung** *(Leader)*  
    Matric No: s299825

    **Pang Hui Ying**  
    Matric No: s295696

    **Wong Jun Hoong**  
    Matric No: s298182

    **Liow Chuan Xuan**  
    Matric No: s298793

    **Cheong Jian Le**  
    Matric No: s301803

    **Lee Jia Heng**  
    Matric No: s302233
    """)




# streamlit run sapp.py

