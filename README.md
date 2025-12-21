# 🚀 Quick Start (For TAs)
Yeseon Hong


## 0. Open `DockerFRTriton`


## 1. Build and Run
We use the `--platform linux/amd64` flag to ensure compatibility with standard server architectures, even if building on Apple Silicon (M1/M2/M3).
### Build the Docker image
```bash
docker build -t fr-triton -f Docker/Dockerfile .
```

### Run the container (Triton + FastAPI)
```bash
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 3000:3000 --name fr_triton fr-triton
```

## 2. Test the API
- Open **http://localhost:3000/docs** in your browser.
- Use the `/face-similarity` endpoint to test with two images.
