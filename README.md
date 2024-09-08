# What is this?

This program performs local image similarity search using Facebook's DINOv2 family of models. 

# Using this program

1. Launch the vector database:

```
sudo docker compose -f compose.yml up
```

2. Install the required packages:

```
pip install -r requirements.txt
```

3. Ingest any files you want to index:

```
python run.py --ingest <path to files> 
```

4. If you don't want to use dinov2-base, you can specify the model size as any of "small", "base", "large", and "giant". E.g.:

```
python run.py --ingest <path to files> --model small
```

5. After that's done, you can search:

```
python run.py --search <path to input image>
```

6. As above, you can specify the model if you don't want to use dinov2-base, e.g.:

```
python run.py --search <path to input image> --model small
```

For additional flags and help, run

```
python run.py --help
```
