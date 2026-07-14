# TensorBoard serving the GCS runs bucket, for Cloud Run.
#
# Reads gs://rl-manipulation-hks-runs directly via gcsfs; on Cloud Run the
# attached service account (tb-viewer, storage.objectViewer on the bucket)
# provides credentials automatically. Deploy from the repo root:
#
#   gcloud run deploy tensorboard --source docker/tensorboard-gcs \
#     --region us-central1 --allow-unauthenticated \
#     --service-account tb-viewer@rl-manipulation-hks.iam.gserviceaccount.com \
#     --memory 1Gi --max-instances 1
FROM python:3.12-slim

RUN pip install --no-cache-dir tensorboard gcsfs

# Rescan the bucket once a minute rather than TensorBoard's aggressive
# default, to keep GCS list operations negligible.
ENV LOGDIR=gs://rl-manipulation-hks-runs

# Shell form so Cloud Run's injected $PORT expands.
CMD exec tensorboard --logdir "$LOGDIR" --bind_all --port "${PORT:-8080}" --reload_interval 60
