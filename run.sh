set -e
touch stop_training.txt;
rm stop_training.txt;
python data_prep.py;
python pretrain_neo.py;
python finetune.py;
uvicorn app:app --host 0.0.0.0 --port 8000&
#curl -X POST "http://localhost:8000/generate-text" -H "Content-Type: application/json" -d '{"prompt": "Prompt: What is the meaning of life?"}';
