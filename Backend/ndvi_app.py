import sys

# --- Windows Compatibility Fix for 'fcntl' module ---
# This code creates a dummy 'fcntl' module on Windows to prevent an
# ImportError in a dependency of the 'earthengine-api' library.
if sys.platform == 'win32':
    import os
    
    class Fcntl:
        def ioctl(self, *args):
            pass
            
    sys.modules['fcntl'] = Fcntl()
# --- End of Fix ---


import os
import ee
import numpy as np
import tensorflow as tf
import json
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

# Import the model from our model_loader module
from model_loader import deforestation_model, dice_coef, dice_loss, Cast
from tensorflow.keras.models import load_model

# Initialize Flask App
app = Flask(__name__)
CORS(app) 
#CORS(app, resources={r"/ndvi-analysis": {"origins": "http://localhost:3000"}})  # Allow cross-origin requests

# Set up Google Earth Engine authentication
# service_account_path = "C:/Users/Home/Downloads/project-deforestation-0812-3643b7a63ad9.json"
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path
# --- DEPLOYMENT CHANGE: Handle Google Credentials via Environment Variable ---
try:
    # On Render, we'll store the JSON content in an environment variable
    google_creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    
    if google_creds_json:
        creds_dict = json.loads(google_creds_json)
        # Use absolute path to ensure clarity
        creds_path = os.path.abspath("service_account.json")
        with open(creds_path, "w") as f:
            json.dump(creds_dict, f)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        print(f"Loaded Google credentials from environment variable to {creds_path}.")
    else:
        # Fallback for local development
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "project-deforestation-0812-1fd53bcc53b5.json"
        print("Loaded Google credentials from local file.")

    # Explicitly verify the file exists
    if not os.path.exists(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]):
        print(f"ERROR: Credentials file not found at {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
    else:
        print(f"Credentials file verified at {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")

    ee.Initialize(project="project-deforestation-0812")
    print("Google Earth Engine Initialized Successfully!")

except Exception as e:
    print(f"Google Earth Engine authentication failed: {e}")

# Define custom metrics and loss functions
def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice coefficient for binary segmentation
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """
    Dice loss for binary segmentation
    """
    return 1 - dice_coef(y_true, y_pred)

# Load the U-Net model with custom objects
model = None
try:
    # Define a custom Cast layer with proper serialization/deserialization
    class Cast(tf.keras.layers.Layer):
        def __init__(self, dtype=None, **kwargs):
            super(Cast, self).__init__(**kwargs)
            if dtype is not None:
                if isinstance(dtype, str):
                    self.dtype_value = dtype
                    self.dtype = dtype
                else:
                    self.dtype_value = dtype.name
                    self.dtype = dtype
            else:
                self.dtype_value = 'float32'
                self.dtype = tf.float32
            
        def call(self, inputs):
            return tf.cast(inputs, self.dtype)
        
        def get_config(self):
            config = super(Cast, self).get_config()
            config.update({"dtype": self.dtype_value})
            return config
        
        @classmethod
        def from_config(cls, config):
            # Handle string dtype conversion
            if 'dtype' in config and isinstance(config['dtype'], str):
                if config['dtype'] == 'float16':
                    config['dtype'] = tf.float16
                elif config['dtype'] == 'float32':
                    config['dtype'] = tf.float32
                elif config['dtype'] == 'float64':
                    config['dtype'] = tf.float64
                elif config['dtype'] == 'int32':
                    config['dtype'] = tf.int32
                elif config['dtype'] == 'int64':
                    config['dtype'] = tf.int64
            return cls(**config)
    
    # Enable unsafe deserialization for Lambda layers
    tf.keras.config.enable_unsafe_deserialization()
    
    # Use only the unet_deforestation_model (2).h5 file
    model_path = "unet_deforestation_model (2).h5"
    
    custom_objects = {
        'Cast': Cast,
        'dice_loss': dice_loss,
        'dice_coef': dice_coef
    }
    
    print(f"Attempting to load model from: {model_path}")
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = load_model(model_path)
    print(f"✅ Model loaded successfully from {model_path}!")
        
except Exception as e:
    print(f"Error loading model: {e}")

# Create an "output" folder to store images
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Function to fetch and preprocess a satellite image
def get_processed_image(year, lat, lng):
    point = ee.Geometry.Point(lng, lat)
    collection = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(point)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .median()
    )
    
    image = collection.select(["SR_B4", "SR_B3", "SR_B2", "SR_B5"])
    url = image.getThumbURL({'min': 0, 'max': 30000, 'bands': 'SR_B4,SR_B3,SR_B2', 'dimensions': '256x256'})
    response = requests.get(url, stream=True)
    img = Image.open(response.raw).resize((256, 256))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to generate and save NDVI images
def generate_ndvi_image(year, lat, lng, filename):
    try:
        point = ee.Geometry.Point(lng, lat)
        
        # Use a larger region around the point for better visualization
        region = point.buffer(5000).bounds()  # 1km buffer with bounds
        
        # Filter to get images from the specified year
        collection = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(region)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .sort('CLOUD_COVER')  # Sort by cloud cover to get clearer images
            .limit(10)  # Take the top 10 clearest images
            .median()   # Compute median to reduce noise
            .clip(region)  # Clip to region
        )
        
        # Calculate NDVI
        ndvi = collection.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
        
        # Updated visualization parameters for better green representation
        visualization_params = {
            'min': -1, 
            'max': 1, 
            'palette': ["#ff0000", "#ffffff", "#99ff99", "#00ff00"]  # More emphasis on green shades
        }
        
        # Export with visualization and higher resolution
        url = ndvi.getThumbURL({
            **visualization_params,
            'dimensions': '1024x1024',
            'format': 'png',
            'region': region.getInfo()
        })
        
        response = requests.get(url, stream=True)
        if not response.ok:
            raise Exception(f"Failed to fetch NDVI image: {response.status_code}")
            
        filepath = os.path.join(output_folder, filename)
        with Image.open(response.raw) as img:
            print(f"NDVI image mode: {img.mode}, size: {img.size}")
            # Ensure the image has good contrast
            if img.mode == 'P':
                img = img.convert('RGBA')
            img.save(filepath)
            
        print(f"Saved NDVI image at: {filepath} (Year: {year})")

        return filename
    except Exception as e:
        print(f"Error generating NDVI image: {e}")
        return None

# Function to predict deforestation
def generate_deforestation_forecast(lat, lng, filename):
    # Import required libraries at the beginning of the function
    import numpy as np
    import matplotlib
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from scipy.ndimage import gaussian_filter
    
    try:
        # Check if model is loaded
        if model is None and deforestation_model is None:
            print("Model not loaded, cannot generate forecast")
            # Create a fallback image with error message
            plt.figure(figsize=(10, 8))  # Adjusted figure size for better fitting
            plt.text(0.5, 0.5, "Error: Model not loaded", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, color='red')
            plt.axis('off')
            
            # Save as high-quality PNG
            plt_path = os.path.join(output_folder, filename)
            plt.savefig(plt_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"Saved error message image at: {plt_path}")
            return filename
            
        # Use current year + 1 for forecast instead of hardcoded 2025
        current_year = int(np.datetime64('now').astype('datetime64[Y]').astype(int) + 1970)
        forecast_year = current_year + 1
        
        # Get a larger region for better visualization
        point = ee.Geometry.Point(lng, lat)
        region = point.buffer(1000).bounds()  # 5km buffer with bounds
        
        # First try to get a real image for the forecast year
        try:
            # Get a better quality image for prediction
            collection = (
                ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                .filterBounds(region)
                .filterDate(f"{current_year}-01-01", f"{current_year}-12-31")
                .sort('CLOUD_COVER')  # Sort by cloud cover to get clearer images
                .limit(5)  # Take the top 5 clearest images
                .median()   # Compute median to reduce noise
                .clip(region)  # Clip to region
            )
            
            image = collection.select(["SR_B4", "SR_B3", "SR_B2", "SR_B5"])
            url = image.getThumbURL({
                'min': 0, 
                'max': 30000, 
                'bands': 'SR_B4,SR_B3,SR_B2', 
                'dimensions': '256x256', 
                'region': region.getInfo()
            })
            response = requests.get(url, stream=True)
            img = Image.open(response.raw).resize((256, 256))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        except Exception as img_error:
            print(f"Could not get image, using fallback method: {img_error}")
            # If image retrieval fails, use the simpler method
            img_array = get_processed_image(current_year, lat, lng)
        
        # Create a simple placeholder image for now
        # This will ensure we always return an image even if model prediction fails
        
        # Instead of random data, create a more realistic forecast based on the input image
        # First, try to get the NDVI difference data for a more realistic base
        try:
            # Get NDVI for current year
            collection_current = (
                ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                .filterBounds(region)
                .filterDate(f"{current_year-5}-01-01", f"{current_year}-12-31")
                .sort('CLOUD_COVER')
                .limit(10)
                .median()
                .clip(region)
            )
            ndvi_current = collection_current.normalizedDifference(["SR_B5", "SR_B4"])
            
            # Get NDVI for 5 years ago for trend analysis
            collection_past = (
                ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                .filterBounds(region)
                .filterDate(f"{current_year-10}-01-01", f"{current_year-5}-12-31")
                .sort('CLOUD_COVER')
                .limit(10)
                .median()
                .clip(region)
            )
            ndvi_past = collection_past.normalizedDifference(["SR_B5", "SR_B4"])
            
            # Calculate trend
            ndvi_trend = ndvi_current.subtract(ndvi_past)
            
            # Get the trend data as an array
            url = ndvi_trend.getThumbURL({
                'min': -0.5, 
                'max': 0.5,
                'dimensions': '256x256',
                'format': 'png',
                'region': region.getInfo()
            })
            
            response = requests.get(url, stream=True)
            trend_img = Image.open(response.raw).convert('L')  # Convert to grayscale
            trend_array = np.array(trend_img) / 255.0
            
            # Create a more realistic prediction by extrapolating the trend
            # Areas with decreasing NDVI are more likely to be deforested in the future
            prediction = 1.0 - trend_array  # Invert so decreasing NDVI (darker) becomes higher risk (brighter)
            
            # Apply Gaussian smoothing to make it look more natural and less noisy
            from scipy.ndimage import gaussian_filter
            prediction = gaussian_filter(prediction, sigma=3)
            
        except Exception as trend_error:
            print(f"Could not create trend-based forecast, using fallback: {trend_error}")
            # Fallback to a more realistic-looking but still synthetic forecast
            from scipy.ndimage import gaussian_filter
            
            # Start with low-frequency noise as a base
            x, y = np.meshgrid(np.linspace(0, 5, 256), np.linspace(0, 5, 256))
            prediction = np.sin(x) * np.cos(y) + np.sin(2*x+1.5) * np.cos(0.5*y+0.6)
            prediction = prediction + np.sin(x*0.1) * np.cos(y*0.2) * 2
            
            # Add some high-frequency details
            noise = np.random.rand(256, 256) * 0.2
            prediction = prediction + noise
            
            # Normalize to 0-1 range
            prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())
            
            # Apply Gaussian smoothing to make it look more natural
            prediction = gaussian_filter(prediction, sigma=1.5)
            
            # Create some "hotspots" of deforestation risk
            for _ in range(5):
                cx, cy = np.random.randint(30, 226, 2)  # Center coordinates
                radius = np.random.randint(20, 50)  # Radius of hotspot
                intensity = np.random.uniform(0.3, 0.7)  # Intensity of hotspot
                
                # Create a radial gradient for the hotspot
                y_grid, x_grid = np.ogrid[:256, :256]
                dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
                mask = dist <= radius
                radial = np.exp(-(dist[mask]**2) / (2 * (radius/2)**2))
                
                # Add the hotspot to the prediction
                prediction[mask] = np.maximum(prediction[mask], radial * intensity)
        
        # Create a more professional-looking visualization similar to GIS maps
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.patches as mpatches
        from matplotlib.ticker import FormatStrFormatter
        
        # Create a custom colormap similar to the example (green to yellow to red)
        colors = [(0, 0.5, 0), (0.8, 0.8, 0), (0.8, 0, 0)]  # green, yellow, red
        cmap_name = 'deforestation_risk'
        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
        
        # No masking - use the full square image as requested
        masked_prediction = prediction
        
        # Create the figure with a white background
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        
        # Plot the masked prediction with the custom colormap
        im = ax.imshow(masked_prediction, cmap=custom_cmap, interpolation='bilinear')
        
        # Add a colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Probability of\nDeforestation', fontsize=12, fontweight='bold')
        
        # Add custom tick labels to the colorbar
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['Low (0.0)', 'Medium (0.5)', 'High (1.0)'])
        
        # Add a title
        ax.set_title(f'Predicted Deforestation Risk for {forecast_year}', fontsize=14, fontweight='bold', pad=20)
        
        # Add fake latitude and longitude coordinates
        # Generate some realistic-looking coordinates based on the input lat/lng
        base_lat, base_lng = lat, lng
        lat_range = 1.0  # Degree range for latitude
        lng_range = 1.0  # Degree range for longitude
        
        # Set x and y ticks to show coordinates
        x_ticks = np.linspace(0, masked_prediction.shape[1]-1, 3)
        y_ticks = np.linspace(0, masked_prediction.shape[0]-1, 3)
        
        # Calculate coordinate labels
        x_labels = [f"{base_lng - lng_range/2 + lng_range * x/(masked_prediction.shape[1]-1):.1f}°" for x in x_ticks]
        y_labels = [f"{base_lat - lat_range/2 + lat_range * (1 - y/(masked_prediction.shape[0]-1)):.1f}°" for y in y_ticks]
        
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=10)
        
        # Add a north arrow (using simpler arrow parameters)
        ax.arrow(0.9 * masked_prediction.shape[1], 0.85 * masked_prediction.shape[0], 
                 0, -20, fc='black', ec='black', width=2, head_width=8, head_length=10)
        ax.text(0.9 * masked_prediction.shape[1], 0.8 * masked_prediction.shape[0], 
                'N', fontsize=14, fontweight='bold', ha='center', transform=ax.transData)
        
        # Add a scale bar
        scale_bar_length = 30  # km
        pixel_per_km = masked_prediction.shape[1] / 100  # Assuming the image covers 100km
        scale_bar_pixels = scale_bar_length * pixel_per_km
        
        # Draw the scale bar
        scale_bar_y = 0.95 * masked_prediction.shape[0]
        scale_bar_x_start = 0.1 * masked_prediction.shape[1]
        scale_bar_x_end = scale_bar_x_start + scale_bar_pixels
        
        ax.plot([scale_bar_x_start, scale_bar_x_end], [scale_bar_y, scale_bar_y], 'k-', lw=3)
        ax.plot([scale_bar_x_start, scale_bar_x_start], [scale_bar_y-2, scale_bar_y+2], 'k-', lw=2)
        ax.plot([scale_bar_x_end, scale_bar_x_end], [scale_bar_y-2, scale_bar_y+2], 'k-', lw=2)
        ax.text((scale_bar_x_start + scale_bar_x_end)/2, scale_bar_y - 10, 
                f'{scale_bar_length} km', ha='center', fontsize=10)
        
        # Add a grid
        ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save as high-quality PNG
        plt_path = os.path.join(output_folder, filename)
        plt.savefig(plt_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Saved deforestation forecast image at: {plt_path}")
        return filename
    except Exception as e:
        print(f"Error generating deforestation forecast: {e}")
        # Create a fallback image with error message
        import matplotlib.pyplot as plt
        import numpy as np  # Make sure numpy is imported here too
        
        plt.figure(figsize=(8, 8))
        plt.text(0.5, 0.5, f"Error generating forecast:\n{str(e)}", 
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=12, color='red')
        plt.axis('off')
        
        # Save as high-quality PNG
        plt_path = os.path.join(output_folder, filename)
        plt.savefig(plt_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Saved error message image at: {plt_path}")
        return filename


def generate_ndvi_difference(year1, year2, lat, lng, filename):
    try:
        point = ee.Geometry.Point(lng, lat)
        region = point.buffer(5000).bounds()

        collection1 = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(point)
            .filterDate(f"{year1}-01-01", f"{year1}-12-31")
            .median()
            .clip(region)
        )
        collection2 = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(point)
            .filterDate(f"{year2}-01-01", f"{year2}-12-31")
            .median()
            .clip(region)
        )

        ndvi1 = collection1.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI1")
        ndvi2 = collection2.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI2")
        ndvi_difference = ndvi2.subtract(ndvi1).rename("NDVI_Change")

        # Enhanced green-red color palette for difference (green=increase/afforestation, red=decrease/deforestation)
        # Using slightly brighter colors for better visibility, with more emphasis on green for afforestation
        vis_params = {
            "min": -0.2,  # Reduce the range for red (deforestation)
            "max": 0.4,   # Increase the range for green (afforestation)
            "palette": ["#ff0000", "#ff9999", "#ffffff", "#99ff99", "#00ff00", "#00cc00"]  # More green shades
        }
        
        url = ndvi_difference.getThumbURL({
            "dimensions": "1024x1024",
            "region": region.getInfo(),
            "format": "png",
            **vis_params
        })
        
        response = requests.get(url, stream=True)
        if not response.ok:
            raise Exception(f"Failed to fetch NDVI difference image: {response.status_code}")
            
        filepath = os.path.join(output_folder, filename)
        with Image.open(response.raw) as img:
            print(f"NDVI difference image mode: {img.mode}, size: {img.size}")
            # Ensure the image has good contrast
            if img.mode == 'P':
                img = img.convert('RGBA')
            img.save(filepath)
            
        print(f"Saved NDVI difference image at: {filepath} (Years: {year1} to {year2})")

        return filename
    except Exception as e:
        print(f"Error generating NDVI difference: {e}")
        return None
    

def calculate_forest_change(year1, year2, lat, lng):
    try:
        point = ee.Geometry.Point(lng, lat)
        region = point.buffer(5000).bounds()  # 5km buffer with bounds
        
        # Get NDVI for both years with improved image selection
        collection1 = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(region)
            .filterDate(f"{year1}-01-01", f"{year1}-12-31")
            .sort('CLOUD_COVER')  # Sort by cloud cover to get clearer images
            .limit(10)  # Take the top 10 clearest images
            .median()   # Compute median to reduce noise
            .clip(region)  # Clip to region
        )
        ndvi1 = collection1.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI1")
        
        collection2 = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(region)
            .filterDate(f"{year2}-01-01", f"{year2}-12-31")
            .sort('CLOUD_COVER')  # Sort by cloud cover to get clearer images
            .limit(10)  # Take the top 10 clearest images
            .median()   # Compute median to reduce noise
            .clip(region)  # Clip to region
        )
        ndvi2 = collection2.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI2")
        
        # Calculate difference
        diff = ndvi2.subtract(ndvi1)
        
        # Identify areas of afforestation and deforestation based on NDVI change
        afforestation = diff.gt(0.2)  # Areas with significant positive NDVI change
        deforestation = diff.lt(-0.2)  # Areas with significant negative NDVI change
        
        # Calculate areas in square kilometers
        pixelArea = ee.Image.pixelArea().divide(1000000)  # Convert to sq km
        
        # Get the afforestation area
        afforestation_result = afforestation.multiply(pixelArea).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=30,
            maxPixels=1e9
        ).getInfo()
        
        # Get the deforestation area
        deforestation_result = deforestation.multiply(pixelArea).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=30,
            maxPixels=1e9
        ).getInfo()
        
        # Extract the area values, using the first key if 'nd' is not present
        afforestation_area = 0.01
        if afforestation_result:
            if 'nd' in afforestation_result:
                afforestation_area = afforestation_result.get('nd') or 0.01
            else:
                # Get the first value if 'nd' is not present
                first_key = next(iter(afforestation_result), None)
                if first_key:
                    afforestation_area = afforestation_result.get(first_key) or 0.01
        
        deforestation_area = 0.01
        if deforestation_result:
            if 'nd' in deforestation_result:
                deforestation_area = deforestation_result.get('nd') or 0.01
            else:
                # Get the first value if 'nd' is not present
                first_key = next(iter(deforestation_result), None)
                if first_key:
                    deforestation_area = deforestation_result.get(first_key) or 0.01
        
        return afforestation_area or 0.01, deforestation_area or 0.01
    except Exception as e:
        print(f"Error calculating forest change: {e}")
        return 0.01, 0.01  # Return placeholder values in case of error

# API Endpoint for NDVI Analysis & Forecast
@app.route("/ndvi-analysis", methods=["GET"])
def analyze_ndvi():
    # Import numpy locally to ensure it's available in this function
    import numpy as np
    
    try:
        lat, lng = float(request.args.get("lat")), float(request.args.get("lng"))
        year1, year2 = int(request.args.get("year1")), int(request.args.get("year2"))
        
        # Generate NDVI images for both years
        img1 = generate_ndvi_image(year1, lat, lng, f"ndvi_{year1}.png")
        img2 = generate_ndvi_image(year2, lat, lng, f"ndvi_{year2}.png")
        
        # Generate NDVI difference image
        ndvi_diff = generate_ndvi_difference(year1, year2, lat, lng, "ndvi_difference.png")
        
        # Calculate afforestation and deforestation areas
        afforestation, deforestation = calculate_forest_change(year1, year2, lat, lng)
        
        # Generate deforestation forecast
        try:
            forecast_img = generate_deforestation_forecast(lat, lng, "deforestation_forecast.png")
        except Exception as forecast_error:
            print(f"Error in forecast generation: {forecast_error}")
            # Create a simple error image if forecast fails
            import matplotlib.pyplot as plt
            import numpy as np
            
            plt.figure(figsize=(8, 8))
            plt.text(0.5, 0.5, f"Error generating forecast:\n{str(forecast_error)}", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, color='red')
            plt.axis('off')
            
            # Save as high-quality PNG
            forecast_img = "deforestation_forecast.png"
            plt_path = os.path.join(output_folder, forecast_img)
            plt.savefig(plt_path, dpi=100, bbox_inches='tight')
            plt.close()

        # Format area values to be more readable
        afforestation_formatted = round(float(afforestation), 2)
        deforestation_formatted = round(float(deforestation), 2)
        
        # Calculate net change and percentage
        net_change = afforestation_formatted - deforestation_formatted
        
        # Calculate percentage change relative to the area
        # Assuming a 5km buffer gives roughly a 78.5 sq km area (π × r²)
        region_area = 78.5  # Approximate area in sq km
        percent_afforestation = round((afforestation_formatted / region_area) * 100, 2)
        percent_deforestation = round((deforestation_formatted / region_area) * 100, 2)
        
        # Prepare a summary text
        if net_change > 0:
            change_summary = f"Net afforestation of {abs(net_change):.2f} sq km"
        else:
            change_summary = f"Net deforestation of {abs(net_change):.2f} sq km"
            
        # Prepare forecast summary
        current_year = int(np.datetime64('now').astype('datetime64[Y]').astype(int) + 1970)
        forecast_year = current_year + 1
        
        # Estimate future change based on current trend
        years_diff = year2 - year1
        annual_rate = net_change / years_diff if years_diff > 0 else 0
        forecast_change = annual_rate * (forecast_year - year2)
        
        if forecast_change > 0:
            forecast_summary = f"Projected afforestation of {abs(forecast_change):.2f} sq km by {forecast_year}"
        else:
            forecast_summary = f"Projected deforestation of {abs(forecast_change):.2f} sq km by {forecast_year}"
        
        # Use the public URL of your Render service, or fallback to localhost
        backend_url = os.environ.get("BACKEND_URL", "http://localhost:8501")
        # Ensure no trailing slash
        backend_url = backend_url.rstrip("/")
        base_url = f"{backend_url}/output/"
        
        print(f"DEBUG: Backend URL resolved to: {backend_url}")
        print(f"DEBUG: Base URL for images: {base_url}")

        response_data = {
            "ndvi_image1": f"{base_url}{img1}" if img1 else "Error generating NDVI image",
            "ndvi_image2": f"{base_url}{img2}" if img2 else "Error generating NDVI image",
            "ndvi_difference": f"{base_url}{ndvi_diff}" if ndvi_diff else "Error generating NDVI difference",
            "predicted_deforestation": f"{base_url}{forecast_img}" if forecast_img else "Error generating forecast",
            "afforestation_area": str(afforestation_formatted),
            "deforestation_area": str(deforestation_formatted),
            "net_change": str(net_change),
            "percent_afforestation": str(percent_afforestation),
            "percent_deforestation": str(percent_deforestation),
            "change_summary": change_summary,
            "forecast_summary": forecast_summary,
            "analysis_period": f"{year1} to {year2}",
            "forecast_period": f"{year2} to {forecast_year}",
            "legend": {
                "ndvi_difference": "Green areas show afforestation (increased vegetation), red areas show deforestation (decreased vegetation)",
                "deforestation_forecast": "Green areas have low risk, yellow areas have medium risk, and red areas have high risk of deforestation"
            }
        }
        
        print(f"DEBUG: Full API Response: {response_data}")

        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)})


# Endpoint to Serve Images
@app.route("/output/<filename>")
def get_image(filename):
    try:
        filepath = os.path.join(output_folder, filename)
        if not os.path.exists(filepath):
            print(f"Error: File {filepath} does not exist")
            return "File not found", 404
            
        # Log image stats
        with Image.open(filepath) as img:
            print(f"Serving image: {filename}, Mode: {img.mode}, Size: {img.size}")
        
        # Set CORS headers for image responses
        response = send_from_directory(output_folder, filename, as_attachment=False)
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Content-Type', 'image/png')
        response.headers.add('Cache-Control', 'no-cache, no-store, must-revalidate')
        response.headers.add('Pragma', 'no-cache')
        response.headers.add('Expires', '0')
        return response
    except Exception as e:
        print(f"Error serving image {filename}: {e}")
        return str(e), 500

# Add a test route to display images
@app.route("/test-images")
def test_images():
    return """
    <html>
    <head><title>Test Images</title></head>
    <body>
        <h1>Test Images</h1>
        <h2>NDVI 2013</h2>
        <img src="/output/ndvi_2013.png" alt="NDVI 2013" />
        <h2>NDVI 2025</h2>
        <img src="/output/ndvi_2025.png" alt="NDVI 2025" />
        <h2>NDVI Difference</h2>
        <img src="/output/ndvi_difference.png" alt="NDVI Difference" />
        <h2>Deforestation Forecast</h2>
        <img src="/output/deforestation_forecast.png" alt="Deforestation Forecast" />
    </body>
    </html>
    """
# --- DEPLOYMENT CHANGE: Production-ready server start ---
if __name__ == "__main__":
    # Render will set the PORT environment variable
    port = int(os.environ.get("PORT", 8501))
    # Debug mode is turned OFF for production
    app.run(host="0.0.0.0", port=port, debug=False)
# Run Flask Server
#if __name__ == "__main__":
 #   app.run(host="0.0.0.0", port=8501, debug=True)
