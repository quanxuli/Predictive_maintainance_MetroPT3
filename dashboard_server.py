import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
from sqlalchemy import create_engine
import os

app = FastAPI(title="Metro-PT3 Dashboard API")

# Setup PostgreSQL Connection
DB_CONNECTION_URL = "postgresql://postgres:anhquan26@localhost:5432/metro_pt3_db"
engine = create_engine(DB_CONNECTION_URL)

# Mount the 'gui' directory to serve static files
current_dir = os.path.dirname(os.path.abspath(__file__))
gui_dir = os.path.join(current_dir, "gui")

# Ensure gui directory exists just in case
if not os.path.exists(gui_dir):
    os.makedirs(gui_dir)

app.mount("/static", StaticFiles(directory=gui_dir), name="static")

@app.get("/")
def serve_gui():
    return FileResponse(os.path.join(gui_dir, "index.html"))

@app.get("/api/data")
def get_dashboard_data(limit: int = 60):
    try:
        # Fetch the most recent 'limit' rows from the database, ordered by timestamp descending
        query = f"""
            SELECT * FROM sensor_realtime_predictions 
            ORDER BY timestamp DESC 
            LIMIT {limit}
        """
        df = pd.read_sql(query, engine)
        if df.empty:
            return {"data": []}
            
        # Reverse the dataframe so oldest is first, newest is last (for charting left to right)
        df = df.iloc[::-1].reset_index(drop=True)
        
        # Convert timestamp to string for JSON serialization
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].astype(str)
            
        return {"data": df.to_dict(orient="records")}
    except Exception as e:
        print(f"Error fetching data: {e}")
        return {"error": str(e), "data": []}

if __name__ == "__main__":
    print("Starting Dashboard Server at http://127.0.0.1:8050")
    uvicorn.run("dashboard_server:app", host="127.0.0.1", port=8050, reload=True)
