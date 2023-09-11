from fastapi import FastAPI


app = FastAPI()


@app.post('/hadiths/semantic_search')
def hadith_semantic_search(query: str) -> list[int]:
    return []


@app.post('/hadiths/count')
def hadith_count() -> int:
    return 0


@app.get('/up')
def up() -> str:
    return 'أنا بخير، شكرا لسؤالك :)'
