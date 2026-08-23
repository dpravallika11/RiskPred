from fastapi import FastAPI

app = FastAPI(
    title="Razorpay Buildathon API",
    description="Backend API for our buildathon MVP",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Backend is running!"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }