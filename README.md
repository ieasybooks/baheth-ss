# البحث بالمعنى

الشيفرة البرمجية لدعم البحث بالمعنى على منصة باحث

## تجهيز بيئة العمل

قم بتنفيذ الأمر التالي لتثبيت مكتبة <code>poetry</code>:

```
pip install poetry
```

ثم قم بتنفيذ الأمر التالي لتثبيت المكتبات المطلوبة:

```
poetry isntall
```

## لتشغيل الخادم المحلي

```
poetry run uvicorn src.baheth_ss.main:app --port 8383 --reload --env-file .env
```

## لبناء وتشغيل Docker image

```
docker build -t baheth_ss-app .
docker run --env-file=.env -d --name baheth_ss -p 8383:8383 baheth_ss-app
```

## لبناء ورفع Docker image

```
docker build -t baheth_ss . --platform linux/amd64
docker image tag baheth_ss ieasybooks/baheth_ss:latest
docker image push ieasybooks/baheth_ss:latest
```
