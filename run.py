import argparse
import hashlib
import mimetypes
import os

from pathlib import Path
from typing import Union

import numpy as np
import psycopg
import torch

from PIL import Image
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

def init_database():
	connection = get_db_connection()
	cursor = connection.cursor()
	cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
	small_create_statement = """
	CREATE TABLE IF NOT EXISTS dino_small(
	file_path TEXT,
	sha3_hash CHAR(64),
	embedding vector(384)
	);"""
	cursor.execute(small_create_statement)

	base_create_statement = """
	CREATE TABLE IF NOT EXISTS dino_base(
	file_path TEXT,
	sha3_hash CHAR(64),
	embedding vector(768)
	);"""
	cursor.execute(base_create_statement)

	large_create_statement = """
	CREATE TABLE IF NOT EXISTS dino_large(
	file_path TEXT,
	sha3_hash CHAR(64),
	embedding vector(1024)
	);"""
	cursor.execute(large_create_statement)

	giant_create_statement = """
	CREATE TABLE IF NOT EXISTS dino_giant(
	file_path TEXT,
	sha3_hash CHAR(64),
	embedding vector(1536)
	);"""
	cursor.execute(giant_create_statement)

	connection.commit()
	cursor.close()

def read_in_chunks(file_object, chunk_size=1024):
	while True:
		data = file_object.read(chunk_size)
		if not data:
			break
		yield data

def hash_file(path: Path, chunk_size: int = 65535) -> str:
	hash_fn = hashlib.sha3_256()
	with open(path, "rb") as f:
		for file_chunk in read_in_chunks(f, chunk_size=chunk_size):
			hash_fn.update(file_chunk)
	return str(hash_fn.hexdigest())

def guess_mime_prefix(path) -> str:
	try:
		prefix = mimetypes.guess_type(path)[0].split("/")[0]
	except Exception as e:
		prefix = ""
	return prefix

def hash_in_database(file_hash: str, model_size: str) -> bool:
	conn = get_db_connection()
	cursor = conn.cursor()
	#search_statement = f"SELECT * FROM dino_{model_size} LIMIT 1;"
	search_statement = f"SELECT * FROM dino_{model_size} WHERE sha3_hash='{file_hash}' LIMIT 1;"
	#print(f"search_statement is {search_statement}")
	cursor.execute(search_statement)
	result = cursor.fetchone()
	if result is None:
		return False
	return True

def get_db_connection():
	db_host = os.environ.get("POSTGRES_HOST", "localhost")
	db_user = os.environ.get("POSTGRES_USER", "user")
	db_name = os.environ.get("POSTGRES_NAME", "vectordb")
	db_port = os.environ.get("POSTGRES_PORT", "5432")
	if db_port[0] != ":":
		db_port = ":" + db_port
	db_password = os.environ.get("POSTGRES_PASSWORD", "password")
	db_url = f"postgresql://{db_user}:{db_password}@{db_host}{db_port}/{db_name}"
	connection = psycopg.connect(db_url)
	register_vector(connection)
	return connection

def embed_image(path: Path, model: AutoModel, processor: AutoImageProcessor) -> Union[np.array, None]:
	device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

	try:
		img = Image.open(path)
	except Exception as e:
		print(f"Error on {path}: {e}")
		return None

	with torch.no_grad():
		inputs = processor(images=img, return_tensors="pt").to(device)
		outputs = model(**inputs)
		image_features = outputs.last_hidden_state
		image_features = image_features.mean(dim=1).numpy().reshape(-1)
	return image_features

def load_model(model_name: str = "facebook/dinov2-base"):
	device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
	processor = AutoImageProcessor.from_pretrained(model_name)
	model = AutoModel.from_pretrained(model_name).to(device)
	return model, processor

def ingest(ingest_path: Path, model_size: str, model: AutoModel, processor: AutoImageProcessor, conn: psycopg.Connection = None) -> None:

	if conn is None:
		conn = get_db_connection()

	cursor = conn.cursor()

	if ingest_path is None:
		ingest_files = set()
	elif ingest_path.is_dir():
		ingest_files = {f for f in ingest_path.rglob("*") if guess_mime_prefix(f) == "image"}
	else:
		ingest_files = {ingest_path}

	for file in tqdm(ingest_files):
		file_hash = hash_file(path=file)
		if hash_in_database(file_hash=file_hash, model_size=model_size):
			continue
		file_embedding = embed_image(path=file, model=model, processor=processor)
		if file_embedding is None:
			continue
		file_path = str(file)
		#print(file_embedding)
		#print(file_embedding.shape)
		#print(type(file_embedding))
		insert_statement = f"INSERT INTO dino_{model_size} (file_path, sha3_hash, embedding) VALUES (%s, %s, %s);"
		cursor.execute(insert_statement, (file_path, file_hash, file_embedding))

	conn.commit()

def search(search_path: Path, search_depth: int, model_size: str, model: AutoModel, processor: AutoImageProcessor, conn: psycopg.Connection = None) -> None:
	if conn is None:
		conn = get_db_connection()

	cursor = conn.cursor()

	file_embedding = embed_image(path=search_path, model=model, processor=processor)
	if file_embedding is None:
		return None
	file_embedding = str(file_embedding.tolist())

	search_string = f"SELECT file_path, (1 - (embedding <=> '{file_embedding}')) AS similarity FROM dino_{model_size} ORDER BY similarity DESC LIMIT {search_depth};"
	cursor.execute(search_string)
	result = cursor.fetchall()
	count = 0
	for r in result:
		count = count + 1
		file_path = r[0]
		similarity = r[1]
		print(f"{count}: {file_path} ({similarity})")


def cli():
	parser = argparse.ArgumentParser()
	parser.add_argument("--ingest", type=Path, help="Path to either a directory or an image, to be indexed and added to the database.")
	parser.add_argument("--search", type=Path, help="Image to use for search.")
	parser.add_argument("--model", choices={"base", "small", "large", "giant"}, help="DINOv2 model to use for embedding and search", default="base")
	parser.add_argument("--search_depth", type=int, help="How many result to return from a search.", default=10)
	args = parser.parse_args()
	ingest_path = args.ingest
	search_path = args.search
	model_size = args.model
	search_depth = args.search_depth

	match model_size:
		case "base":
			model_name = "facebook/dinov2-base"
		case "small":
			model_name = "facebook/dinov2-small"
		case "large":
			model_name = "facebook/dinov2-large"
		case "giant":
			model_name = "facebook/dinov2-giant"
		case _:
			raise ValueError("Model name not recognized.")

	conn = get_db_connection()
	model, processor = load_model(model_name=model_name)

	ingest(ingest_path=ingest_path, model_size=model_size, model=model, processor=processor, conn=conn)

	search(search_path=search_path, search_depth=search_depth, model_size=model_size, model=model, processor=processor, conn=conn)


if __name__ == '__main__':
	load_dotenv()
	init_database()
	cli()
