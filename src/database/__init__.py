# Database Integration Package
# This package will contain modules for interacting with databases,
# such as MongoDB, PostgreSQL, etc.

# from .mongodb_client import MongoDBClient

# Example configuration (could be loaded from main config)
DB_CONFIG = {
    "type": "mongodb", # 'mongodb', 'postgresql', 'sqlite'
    "mongodb": {
        "connection_string": "mongodb://localhost:27017/", # Placeholder, use env var for production
        "database_name": "nexaris_cle_data",
        "default_collection": "sessions"
    },
    "sqlite": {
        "db_path": "data/local_database.db"
    }
}

def get_db_config():
    return DB_CONFIG

def get_database_client(config=None):
    """
    Factory function to get a database client based on configuration.
    """
    if config is None:
        config = get_db_config()
    
    db_type = config.get("type", "sqlite") # Default to sqlite if not specified

    if db_type == "mongodb":
        # from .mongodb_client import MongoDBClient
        # mongo_conf = config.get("mongodb", {})
        # return MongoDBClient(
        #     connection_string=mongo_conf.get("connection_string"),
        #     database_name=mongo_conf.get("database_name")
        # )
        print("[DB] MongoDB client selected (placeholder). Ensure 'pymongo' is installed.")
        return None # Placeholder for actual client
    elif db_type == "sqlite":
        # Potentially a simple SQLite wrapper here or use directly
        print("[DB] SQLite selected. Direct usage or a simple wrapper can be implemented.")
        return None # Placeholder
    else:
        raise ValueError(f"Unsupported database type: {db_type}")