# Example Sensor Plugin

from .sensor import ExampleSensor

# This function could be called by the main plugin loader
def register(register_fn):
    register_fn("ExampleSensor", ExampleSensor)
    print("[ExampleSensor Plugin] Registered.")

# You might also have functions to initialize or get plugin metadata
def get_plugin_info():
    return {
        "name": "Example Sensor Plugin",
        "version": "0.1.0",
        "description": "A sample plugin demonstrating sensor integration.",
        "author": "NEXARIS AI"
    }