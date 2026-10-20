# Deploying the public demo

This document covers running the service in a container locally, and standing it up as a
**public demo deployment** on Azure Container Apps with continuous deployment from `main`.
It is a demo: one replica that scales to zero, no authentication, no traffic beyond the
author's own runs. Do not call it production.

## 1. Run the container locally

```bash
make docker-build                       # builds ghcr.io/dayav/tabular-synth-eval-agent:local
make docker-run                         # reads ANTHROPIC_API_KEY from your shell or .env, serves on :8000
curl -s localhost:8000/health | python -m json.tool
curl -s -F real=@examples/real_sample.csv -F synthetic=@examples/synth_sample.csv \
     -F target_column=target localhost:8000/evaluate | python -m json.tool
```

Run folders land in the container's `/data/runs`. `make docker-run` mounts the repository's
`runs/` directory there so they survive the container.

The image is built by [`Dockerfile`](../Dockerfile): a builder stage installs the pinned
dependencies from `requirements.lock` into a virtualenv; the runtime stage is `python:3.12-slim`
plus `libgomp1` (for xgboost), a non-root user, a `HEALTHCHECK` on `/health`, and only `src/`.
The API key is read from the environment at request time and is never baked into the image.

## 2. What CI does

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) has four jobs:

| job | when | what |
|---|---|---|
| `test` | every PR and push | ruff, mypy, unit tests, regression suite, API tests. No key needed. |
| `evals` | push to `main`, or manual | live eval. On `main`: `claude-haiku-5-5`, one run per case, Opus judge (cents). Manual trigger lets you pick the model, runs and judge. Results are uploaded as an artifact and the comparison table goes to the job summary. Skipped with a warning if the key secret is missing. |
| `docker` | push to `main` | builds the image, pushes `ghcr.io/<owner>/<repo>:<sha>` and `:latest`, smoke-tests `/health` on the pushed image. |
| `deploy` | push to `main` and `vars.AZURE_DEPLOY == 'true'` | logs in to Azure with OIDC, updates the container app's secret and image, waits for `/health`. |

Repository **secret** required for `evals` and `deploy`: `ANTHROPIC_API_KEY`.

## 3. One-time Azure setup (az CLI)

Prerequisites: an Azure subscription, `az` 2.60 or later, the `containerapp` extension
(`az extension add --name containerapp --upgrade`), and the GHCR package set to **public**
visibility (GitHub: package settings, "Change visibility") so Azure can pull it without a
registry credential. Pick names once:

```bash
export RG=tabular-synth-eval-demo
export LOC=canadacentral
export ENV_NAME=tse-env
export APP=tabular-synth-eval-agent
export IMAGE=ghcr.io/dayav/tabular-synth-eval-agent:latest
```

### 3.1 Resource group and Container Apps environment

```bash
az group create --name "$RG" --location "$LOC"
az containerapp env create --name "$ENV_NAME" --resource-group "$RG" --location "$LOC"
```

### 3.2 The container app

```bash
az containerapp create \
  --name "$APP" --resource-group "$RG" --environment "$ENV_NAME" \
  --image "$IMAGE" \
  --target-port 8000 --ingress external \
  --cpu 1.0 --memory 2.0Gi \
  --min-replicas 0 --max-replicas 1 \
  --secrets anthropic-api-key="$ANTHROPIC_API_KEY" \
  --env-vars ANTHROPIC_API_KEY=secretref:anthropic-api-key RUNS_DIR=/data/runs \
             API_DEFAULT_MODEL=claude-haiku-5-5 MAX_UPLOAD_MB=10 MAX_CONCURRENT_EVALUATIONS=1

az containerapp show --name "$APP" --resource-group "$RG" \
  --query properties.configuration.ingress.fqdn -o tsv
curl -s "https://$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)/health"
```

Notes on those choices:

- `--min-replicas 0`: the demo scales to zero and costs nothing idle; the first request after
  idle takes 10 to 20 seconds to cold-start.
- 1 vCPU / 2 GiB: XGBoost on a few thousand rows is fine; raise if you raise `MAX_ROWS`.
- `API_DEFAULT_MODEL=claude-haiku-5-5` keeps a public endpoint cheap if someone finds it.
  `MAX_CONCURRENT_EVALUATIONS=1` caps parallel spend; extra requests get a 503.
- The key goes in as a Container Apps secret and is referenced by `secretref:`; it is never
  in the image, the workflow log, or `az containerapp show` output.

### 3.3 Optional: persistent run logs

Without this, run folders live in the replica's filesystem and vanish when it scales to zero.
For a demo that is acceptable; to keep them:

```bash
export STG=tsedemostorage$RANDOM
az storage account create --name "$STG" --resource-group "$RG" --location "$LOC" --sku Standard_LRS
az storage share create --name runs --account-name "$STG"
KEY=$(az storage account keys list --account-name "$STG" --resource-group "$RG" --query '[0].value' -o tsv)
az containerapp env storage set --name "$ENV_NAME" --resource-group "$RG" \
  --storage-name runs --azure-file-account-name "$STG" --azure-file-account-key "$KEY" \
  --azure-file-share-name runs --access-mode ReadWrite
# then add a volume + volumeMount at /data/runs to the app:
az containerapp show -n "$APP" -g "$RG" -o yaml > app.yaml
#   under template: add  volumes: [{name: runs, storageType: AzureFile, storageName: runs}]
#   under the container: volumeMounts: [{volumeName: runs, mountPath: /data/runs}]
az containerapp update -n "$APP" -g "$RG" --yaml app.yaml
```

### 3.4 Let GitHub Actions deploy (OIDC, no stored Azure password)

```bash
SUB=$(az account show --query id -o tsv)
APP_ID=$(az ad app create --display-name "tse-github-deploy" --query appId -o tsv)
az ad sp create --id "$APP_ID"
az role assignment create --assignee "$APP_ID" --role Contributor --scope "/subscriptions/$SUB/resourceGroups/$RG"
az ad app federated-credential create --id "$APP_ID" --parameters '{
  "name": "github-main",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:dayav/tabular-synth-eval-agent:environment:demo",
  "audiences": ["api://AzureADTokenExchange"]
}'
echo "AZURE_CLIENT_ID=$APP_ID"; echo "AZURE_TENANT_ID=$(az account show --query tenantId -o tsv)"; echo "AZURE_SUBSCRIPTION_ID=$SUB"
```

Then in the GitHub repository:

| kind | name | value |
|---|---|---|
| environment `demo` secret | `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID` | from the commands above |
| repository secret | `ANTHROPIC_API_KEY` | the key the demo should spend from (a dedicated key with a spend limit) |
| repository variable | `AZURE_DEPLOY` | `true` |
| repository variable | `AZURE_RESOURCE_GROUP` | `$RG` |
| repository variable | `AZURE_CONTAINERAPP` | `$APP` |

With those in place, every push to `main` that passes the test gate builds the image, pushes
it to GHCR, updates the container app to the new SHA-tagged image, and waits for `/health`.
Leave `AZURE_DEPLOY` unset and the job is skipped; the image is still published.

## 4. Tear down

```bash
az group delete --name "$RG" --yes --no-wait
```

## 5. Status

As of 2026-10-20 the Dockerfile and workflow are in the repository and the image builds and
serves locally. The Azure resources have **not** been created from this environment (no `az`
login here); section 3 is the exact sequence to run. Until then, the README must not claim a
live URL.
