# Lambda container image for the watershed extraction API.
# Build context is the api/ directory:
#   docker build -t extract-watershed-api api/
FROM public.ecr.aws/lambda/python:3.12

# System libraries required by rasterio (GDAL)
RUN dnf install -y expat && dnf clean all

# Dependencies are installed from a fully pinned requirements.txt generated
# from uv.lock. --only-binary :all: ensures any dependency without a wheel
# (that would fall back to source build) is caught at build time
# (avoid building inside the Lambda image).
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir --only-binary :all: -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Application code (watershed package)
COPY src/watershed ${LAMBDA_TASK_ROOT}/watershed

CMD ["watershed.service.handler.lambda_handler"]
