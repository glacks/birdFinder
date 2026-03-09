# Bird Finder

`data_raw/` 이미지 데이터셋으로 새 종 분류 모델을 학습하고, 웹에서 이미지를 업로드해 예측하는 프로젝트입니다.

## Stack
- Model training/inference: PyTorch + torchvision
- Backend API: FastAPI
- Frontend: React (Vite)

## Project Structure
```txt
.
├─ backend
│  ├─ app
│  │  ├─ main.py
│  │  └─ model.py
│  ├─ artifacts/            # 학습 결과 저장 위치 (생성됨)
│  ├─ requirements.txt
│  └─ train.py
├─ data_raw/                # 클래스별 이미지 폴더 (git ignored)
└─ frontend
   ├─ src
   ├─ package.json
   └─ vite.config.js
```

## 1) Backend Setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) GPU 학습 실행
`cuda`가 있으면 자동으로 GPU를 사용합니다.

```bash
cd backend
source .venv/bin/activate
python train.py \
  --dataset-dir ../data_raw \
  --artifacts-dir artifacts \
  --epochs 8 \
  --batch-size 32 \
  --workers 4
```

학습이 완료되면 아래 파일이 생성됩니다.
- `backend/artifacts/best_model.pt`
- `backend/artifacts/class_names.json`

## 3) Backend API 실행
```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

헬스체크:
```bash
curl http://localhost:8000/health
```

## 4) Frontend 실행
```bash
cd frontend
npm install
npm run dev
```

브라우저에서 `http://localhost:5173` 접속 후 이미지를 업로드하면 예측됩니다.

## API
### `POST /predict`
- form-data 키: `file` (image)
- 응답 예시:
```json
{
  "top_prediction": {
    "class_name": "Barn_Swallow",
    "probability": 0.872341
  },
  "top_k": [
    { "class_name": "Barn_Swallow", "probability": 0.872341 },
    { "class_name": "Bank_Swallow", "probability": 0.045993 }
  ]
}
```
