import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_db_path = os.environ.get("DATABASE_PATH", os.path.join(BASE_DIR, "demosite.db"))
DATABASE_URL = f"sqlite:///{_db_path}"
APP_NAME = "DemoSite – Phase 1"
DEBUG = True
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
