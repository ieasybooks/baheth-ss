from fastapi import FastAPI


app = FastAPI()


@app.post('/hadith_ss')
def hadith_ss(query: str) -> list[int]:
    return []


@app.get('/up')
def up() -> str:
    return 'أنا بخير، شكرا لسؤالك :)'
