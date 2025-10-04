import tensorflow as tf
import os

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

# Define a custom Cast layer with proper serialization/deserialization
class Cast(tf.keras.layers.Layer):
    def __init__(self, dtype=None, **kwargs):
        super(Cast, self).__init__(**kwargs)
        self._set_dtype(dtype)
        
    def _set_dtype(self, dtype):
        if dtype is not None:
            if isinstance(dtype, str):
                self.dtype_value = dtype
                if dtype == 'float16':
                    self.dtype = tf.float16
                elif dtype == 'float32':
                    self.dtype = tf.float32
                elif dtype == 'float64':
                    self.dtype = tf.float64
                elif dtype == 'int32':
                    self.dtype = tf.int32
                elif dtype == 'int64':
                    self.dtype = tf.int64
                else:
                    self.dtype = dtype
            else:
                self.dtype_value = dtype.name if hasattr(dtype, 'name') else str(dtype)
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
        # Create a new instance
        instance = cls(**config)
        return instance

def load_deforestation_model():
    """
    Load the deforestation prediction model with custom objects
    """
    # Enable unsafe deserialization for Lambda layers
    tf.keras.config.enable_unsafe_deserialization()
    
    # Try different model files in order of preference
    model_files = [
        "unet_deforestation_model (2).h5",
        "unet2_deforestation_model.keras",
        "unet2_deforestation_model (1).keras",
        "unet_deforestation_model.keras"
    ]
    
    custom_objects = {
        'Cast': Cast,
        'dice_loss': dice_loss,
        'dice_coef': dice_coef,
        'cast_to_float32': lambda x: tf.cast(x, tf.float32)
    }
    
    # Try loading each model file until one succeeds
    for model_path in model_files:
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found, trying next...")
            continue
            
        print(f"Attempting to load model from: {model_path}")
        try:
            with tf.keras.utils.custom_object_scope(custom_objects):
                model = tf.keras.models.load_model(model_path)
            print(f"✅ Model loaded successfully from {model_path}!")
            return model
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
    
    # If we get here, all model loading attempts failed
    print("❌ Failed to load any model file!")
    return None

# Global model instance that can be imported
deforestation_model = load_deforestation_model()
