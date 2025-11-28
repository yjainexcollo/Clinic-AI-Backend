# Azure AI Foundry Integration Guide for Clinic-AI

**Version**: 1.0  
**Last Updated**: November 27, 2025  
**Status**: ✅ Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [What is Azure AI Foundry?](#what-is-azure-ai-foundry)
3. [How Azure AI Foundry Replaces Helicone](#how-azure-ai-foundry-replaces-helicone)
4. [Architecture](#architecture)
5. [Required Azure Resources](#required-azure-resources)
6. [Environment Variables](#environment-variables)
7. [Setup Guide](#setup-guide)
8. [Validation and Testing](#validation-and-testing)
9. [Monitoring and Observability](#monitoring-and-observability)
10. [KQL Query Library](#kql-query-library)
11. [Troubleshooting](#troubleshooting)
12. [Cost Optimization](#cost-optimization)
13. [Best Practices](#best-practices)

---

## Overview

Clinic-AI has completed migration from Helicone to **Azure AI Foundry** for comprehensive monitoring of all AI operations. This guide provides everything you need to understand, deploy, and maintain Azure AI Foundry integration.

### Key Benefits

- ✅ **100% Azure Native** - No third-party dependencies
- ✅ **Zero Code Changes** - Automatic log capture
- ✅ **Real-Time Monitoring** - Live metrics and traces
- ✅ **Cost Efficiency** - No per-request fees
- ✅ **HIPAA Compliant** - Enterprise-grade security
- ✅ **Unified Observability** - Single pane of glass

---

## What is Azure AI Foundry?

Azure AI Foundry (formerly Azure AI Studio) is Microsoft's comprehensive platform for:

- **AI Model Management** - Deploy and version control AI models
- **Observability** - Monitor token usage, latency, errors
- **Cost Analytics** - Track spending per deployment
- **Prompt Management** - Version and test prompts
- **Evaluation** - Assess model performance
- **Integration** - Connect with Azure OpenAI, Application Insights, Log Analytics

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                   Azure AI Foundry Workspace                 │
│                                                              │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Deployments  │  │   Monitoring    │  │  Evaluation  │ │
│  │   Management   │  │   Dashboard     │  │  & Testing   │ │
│  └────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
        │                      │                      │
        ├──────────────────────┼──────────────────────┤
        │                      │                      │
┌───────▼──────┐  ┌───────────▼────────┐  ┌─────────▼─────────┐
│ Azure OpenAI │  │ Log Analytics      │  │ App Insights      │
│ Resource     │  │ Workspace          │  │                   │
└──────────────┘  └────────────────────┘  └───────────────────┘
```

---

## How Azure AI Foundry Replaces Helicone

### Comparison Matrix

| Feature | Helicone | Azure AI Foundry | Winner |
|---------|----------|------------------|--------|
| **Setup Complexity** | Medium (proxy config) | Low (zero-code) | ✅ Foundry |
| **Latency Impact** | +50-100ms (proxy hop) | 0ms (native) | ✅ Foundry |
| **Cost** | $0.002 per 1K requests | Included | ✅ Foundry |
| **Data Privacy** | Third-party | Azure-native | ✅ Foundry |
| **Token Tracking** | ✅ Yes | ✅ Yes | Tie |
| **Latency Metrics** | ✅ Yes | ✅ Yes | Tie |
| **Error Tracking** | ✅ Yes | ✅ Yes | Tie |
| **Cost Analytics** | ✅ Yes | ✅ Yes | Tie |
| **Custom Dashboards** | Limited | Full KQL | ✅ Foundry |
| **Alerting** | Basic | Advanced | ✅ Foundry |
| **HIPAA Compliance** | Requires BAA | Native | ✅ Foundry |
| **Integration** | API | Native Azure | ✅ Foundry |

### Migration Benefits

1. **🚀 Performance**: Removed proxy latency (50-100ms saved per request)
2. **💰 Cost Savings**: $20-50/month saved on Helicone fees
3. **🔒 Security**: PHI stays within Azure boundary
4. **📊 Insights**: Richer analytics via Log Analytics
5. **🔗 Integration**: Unified with Azure App Service monitoring

---

## Architecture

### High-Level Flow

```
┌──────────────────────────────────────────────────────────────┐
│                        Clinic-AI Backend                      │
│                                                               │
│  FastAPI Application                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Service Layer                                       │   │
│  │  ├─ OpenAISoapService (SOAP generation)            │   │
│  │  ├─ OpenAIQuestionService (adaptive intake)        │   │
│  │  └─ AzureSpeechService (transcription)             │   │
│  └────────────────┬────────────────────────────────────┘   │
│                   │                                          │
│  ┌────────────────▼────────────────────────────────────┐   │
│  │  AzureAIClient (core/ai_client.py)                  │   │
│  │  - Direct AsyncAzureOpenAI calls                    │   │
│  │  - No proxy, no middleware                          │   │
│  │  - Request ID tracking                               │   │
│  └────────────────┬────────────────────────────────────┘   │
└───────────────────┼─────────────────────────────────────────┘
                    │
                    │ Native Azure OpenAI SDK calls
                    ▼
┌──────────────────────────────────────────────────────────────┐
│                    Azure OpenAI Service                       │
│  ┌──────────────────────────────────────────────────┐       │
│  │  Deployment: gpt-4o-mini                         │       │
│  │  API Version: 2025-01-01-preview                 │       │
│  │  Endpoint: https://clinicai-openai.openai.azure.com │     │
│  └─────────────────┬────────────────────────────────┘       │
└────────────────────┼───────────────────────────────────────────┘
                     │ Automatic Diagnostic Logs
                     ▼
┌──────────────────────────────────────────────────────────────┐
│              Log Analytics Workspace                          │
│  ┌──────────────────────────────────────────────────┐       │
│  │  Tables:                                          │       │
│  │  - AzureOpenAIRequests (prompts, responses)      │       │
│  │  - AzureOpenAIUsage (token counts)               │       │
│  │  - AzureOpenAITraces (latency, errors)           │       │
│  │  - AppTraces (Application Insights)              │       │
│  │  - AppRequests (HTTP requests)                    │       │
│  └──────────────────────────────────────────────────┘       │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│              Azure AI Foundry Monitoring                      │
│  - Token usage dashboard                                     │
│  - Latency analytics                                         │
│  - Error tracking                                            │
│  - Cost analysis                                             │
│  - Request correlation                                       │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Request Initiation**: Clinic-AI service calls `AzureAIClient.chat()`
2. **Azure OpenAI Call**: Native `AsyncAzureOpenAI` SDK makes HTTPS request
3. **Automatic Logging**: Azure OpenAI diagnostics log to Log Analytics
4. **Application Insights**: FastAPI telemetry via OpenTelemetry
5. **Foundry Aggregation**: AI Foundry queries Log Analytics for dashboards

---

## Required Azure Resources

### 1. Azure OpenAI Resource

**Purpose**: Host GPT models for LLM operations

**SKU**: Standard (S0)

**Configuration**:
- Deployment: `gpt-4o-mini` (single deployment used by all workloads)
- Provisioned Throughput: Optional (for guaranteed capacity)
- Network Access: Public (or private endpoint for production)
- Managed Identity: Enable for keyless access (optional)

**Cost**: Pay-per-token (~$0.15 per 1M input tokens, $0.60 per 1M output tokens for gpt-4o-mini)

### 2. Azure AI Foundry Workspace

**Purpose**: Central monitoring and management hub

**SKU**: Standard

**Linked Services**:
- Azure OpenAI resource (created above)
- Log Analytics Workspace
- Application Insights
- Azure Storage Account (for artifacts)

**Cost**: Free (pay only for underlying resources)

### 3. Log Analytics Workspace

**Purpose**: Store diagnostic logs and query with KQL

**SKU**: Pay-as-you-go

**Retention**: 30 days (default), extend to 730 days for compliance

**Data Sources**:
- Azure OpenAI diagnostic logs
- Application Insights telemetry
- Azure Speech Service logs (optional)

**Cost**: ~$2.76 per GB ingested + retention fees

### 4. Application Insights

**Purpose**: Application-level telemetry (HTTP requests, dependencies)

**SKU**: Pay-as-you-go

**Instrumentation**: OpenTelemetry via `azure-monitor-opentelemetry`

**Features**:
- Request tracing
- Dependency tracking
- Exception monitoring
- Custom metrics

**Cost**: $2.88 per GB ingested (first 5GB/month free)

### 5. Azure Key Vault (Recommended)

**Purpose**: Secure secret management

**SKU**: Standard

**Secrets to Store**:
- `AZURE-OPENAI-API-KEY`
- `AZURE-OPENAI-ENDPOINT`
- `MONGO-URI`
- `SECURITY-SECRET-KEY`
- `AZURE-SPEECH-SUBSCRIPTION-KEY`

**Cost**: $0.03 per 10,000 operations

### 6. Azure App Service

**Purpose**: Host Clinic-AI backend

**SKU**: P1V2 or higher (for production)

**Configuration**:
- Runtime: Python 3.11
- Always On: Enabled
- ARR Affinity: Disabled (for stateless apps)
- Health Check: `/health/live`

**Cost**: ~$73/month (P1V2)

---

## Environment Variables

### Complete Configuration

```bash
# ============================================================================
# AZURE OPENAI CONFIGURATION (REQUIRED)
# ============================================================================

# Azure OpenAI endpoint (WITHOUT trailing slash)
AZURE_OPENAI_ENDPOINT=https://clinicai-openai.openai.azure.com

# Azure OpenAI API key (from Azure Portal > Keys and Endpoint)
AZURE_OPENAI_API_KEY=your_api_key_here

# API version - use latest preview for newest features
AZURE_OPENAI_API_VERSION=2025-01-01-preview

# Chat deployment name (must match deployment in Azure Portal)
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini

# ============================================================================
# AZURE SPEECH SERVICE (REQUIRED for transcription)
# ============================================================================

# Azure Speech subscription key
AZURE_SPEECH_SUBSCRIPTION_KEY=your_speech_key_here

# Azure region (e.g., eastus, westus2, centralus)
AZURE_SPEECH_REGION=eastus

# Enable speaker diarization (Doctor/Patient separation)
AZURE_SPEECH_ENABLE_SPEAKER_DIARIZATION=true

# Maximum speakers to identify
AZURE_SPEECH_MAX_SPEAKERS=2

# Transcription mode (batch recommended for accuracy)
AZURE_SPEECH_TRANSCRIPTION_MODE=batch

# Polling interval for batch status (seconds)
AZURE_SPEECH_BATCH_POLLING_INTERVAL=5

# Maximum wait time for batch transcription (seconds)
AZURE_SPEECH_BATCH_MAX_WAIT_TIME=1800

# ============================================================================
# AZURE APPLICATION INSIGHTS (RECOMMENDED)
# ============================================================================

# Application Insights connection string (from Azure Portal)
APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=xxxx;IngestionEndpoint=https://eastus-8.in.applicationinsights.azure.com/;LiveEndpoint=https://eastus.livediagnostics.monitor.azure.com/

# ============================================================================
# AZURE KEY VAULT (OPTIONAL - for production)
# ============================================================================

# Key Vault name (secrets auto-loaded if configured)
AZURE_KEY_VAULT_NAME=your-key-vault-name

# ============================================================================
# MONGODB CONFIGURATION
# ============================================================================

MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/
MONGO_DB_NAME=clinicai

# ============================================================================
# APPLICATION SETTINGS
# ============================================================================

APP_ENV=production
DEBUG=false
PORT=8000
HOST=0.0.0.0

# ============================================================================
# SECURITY
# ============================================================================

SECURITY_SECRET_KEY=your_secret_key_here_must_be_at_least_32_characters_long
API_KEYS=key1:user1,key2:user2

# ============================================================================
# AZURE BLOB STORAGE (for audio files)
# ============================================================================

AZURE_BLOB_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=xxx;AccountKey=xxx;EndpointSuffix=core.windows.net
AZURE_BLOB_CONTAINER_NAME=clinicaiblobstorage
```

### Critical Variables for Azure AI Foundry

| Variable | Purpose | Impact on Foundry |
|----------|---------|-------------------|
| `AZURE_OPENAI_ENDPOINT` | Defines which Azure OpenAI resource to use | Logs go to this resource's diagnostic settings |
| `AZURE_OPENAI_API_VERSION` | API version for OpenAI calls | Must be compatible with deployment |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Which model deployment to use | Foundry tracks usage per deployment |
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | Links to App Insights | Enables correlated tracing |

---

## Setup Guide

### Step 1: Create Azure OpenAI Resource

#### Via Azure Portal

1. Navigate to **Azure Portal** → **Create a resource**
2. Search for "Azure OpenAI"
3. Click **Create**
4. Fill in:
   - **Subscription**: Your subscription
   - **Resource Group**: `clinic-ai-prod` (or create new)
   - **Region**: `East US` (or preferred region)
   - **Name**: `clinicai-openai`
   - **Pricing tier**: `Standard S0`
5. Click **Review + Create** → **Create**

#### Via Azure CLI

```bash
# Set variables
RESOURCE_GROUP="clinic-ai-prod"
LOCATION="eastus"
OPENAI_NAME="clinicai-openai"

# Create resource group
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION

# Create Azure OpenAI resource
az cognitiveservices account create \
  --name $OPENAI_NAME \
  --resource-group $RESOURCE_GROUP \
  --kind OpenAI \
  --sku S0 \
  --location $LOCATION
```

### Step 2: Deploy GPT Model

#### Via Azure Portal

1. Go to your Azure OpenAI resource
2. Click **Model deployments** → **Create**
3. Select model: `gpt-4o-mini`
4. Name deployment: `gpt-4o-mini`
5. Set capacity: `10K TPM` (tokens per minute)
6. Click **Create**

#### Via Azure CLI

```bash
# Deploy gpt-4o-mini
az cognitiveservices account deployment create \
  --name $OPENAI_NAME \
  --resource-group $RESOURCE_GROUP \
  --deployment-name gpt-4o-mini \
  --model-name gpt-4o-mini \
  --model-version "2024-07-18" \
  --model-format OpenAI \
  --sku-capacity 10 \
  --sku-name "Standard"
```

### Step 3: Enable Diagnostic Logging

#### Via Azure Portal

1. Go to Azure OpenAI resource
2. Click **Diagnostic settings** → **Add diagnostic setting**
3. Name: `foundry-diagnostics`
4. Select logs:
   - ✅ `Audit`
   - ✅ `RequestResponse`
   - ✅ `Trace`
5. Destination:
   - ✅ **Send to Log Analytics workspace**
   - Select your Log Analytics workspace
6. Click **Save**

#### Via Azure CLI

```bash
# Get OpenAI resource ID
OPENAI_ID=$(az cognitiveservices account show \
  --name $OPENAI_NAME \
  --resource-group $RESOURCE_GROUP \
  --query id -o tsv)

# Get Log Analytics workspace ID
WORKSPACE_ID=$(az monitor log-analytics workspace show \
  --resource-group $RESOURCE_GROUP \
  --workspace-name clinic-ai-logs \
  --query id -o tsv)

# Create diagnostic setting
az monitor diagnostic-settings create \
  --name foundry-diagnostics \
  --resource $OPENAI_ID \
  --workspace $WORKSPACE_ID \
  --logs '[
    {"category": "Audit", "enabled": true},
    {"category": "RequestResponse", "enabled": true},
    {"category": "Trace", "enabled": true}
  ]'
```

### Step 4: Create AI Foundry Workspace

#### Via Azure Portal

1. Navigate to **AI Foundry** (formerly AI Studio): https://ai.azure.com
2. Click **Create workspace**
3. Fill in:
   - **Name**: `clinic-ai-foundry`
   - **Subscription**: Your subscription
   - **Resource group**: `clinic-ai-prod`
   - **Region**: Same as Azure OpenAI
4. **Link resources**:
   - Azure OpenAI: Select your resource
   - Log Analytics: Select your workspace
   - Application Insights: Select your resource
5. Click **Create**

#### Via Azure CLI

```bash
# Create AI Foundry workspace (using Azure ML workspace as foundation)
az ml workspace create \
  --name clinic-ai-foundry \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --application-insights /subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Insights/components/clinic-ai-insights
```

### Step 5: Configure Application Insights

#### Install Python packages (already in requirements.txt)

```bash
pip install azure-monitor-opentelemetry opentelemetry-instrumentation-fastapi
```

#### Code integration (already done in `app.py`)

```python
# In app.py
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Configure Azure Monitor
configure_azure_monitor(
    connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
)

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)
```

### Step 6: Deploy to Azure App Service

#### Create App Service

```bash
# Set variables
APP_NAME="clinic-ai-backend"
APP_SERVICE_PLAN="clinic-ai-plan"

# Create App Service Plan
az appservice plan create \
  --name $APP_SERVICE_PLAN \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku P1V2 \
  --is-linux

# Create Web App
az webapp create \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --plan $APP_SERVICE_PLAN \
  --runtime "PYTHON:3.11"
```

#### Configure Environment Variables

```bash
# Set all environment variables
az webapp config appsettings set \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --settings \
    AZURE_OPENAI_ENDPOINT="https://clinicai-openai.openai.azure.com" \
    AZURE_OPENAI_API_KEY="@Microsoft.KeyVault(SecretUri=https://your-keyvault.vault.azure.net/secrets/AZURE-OPENAI-API-KEY/)" \
    AZURE_OPENAI_API_VERSION="2025-01-01-preview" \
    AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o-mini" \
    APPLICATIONINSIGHTS_CONNECTION_STRING="InstrumentationKey=xxx;..." \
    MONGO_URI="@Microsoft.KeyVault(SecretUri=...)" \
    MONGO_DB_NAME="clinicai"
```

---

## Validation and Testing

### Test 1: Verify Basic Connectivity

```bash
# Run verification script (from backend directory)
cd /Users/excollodev/Desktop/Clinic-AI/backend
python3 scripts/verify_foundry.py
```

**Note**: The script automatically adds the `src` directory to the Python path, so you can run it directly from the backend directory.

Expected output:
```
🔍 Verifying Azure AI Foundry integration...

✅ Client initialized successfully
📍 Endpoint: https://clinicai-openai.openai.azure.com
🔖 API Version: 2025-01-01-preview
🤖 Deployment: gpt-4o-mini

🚀 Sending test request to Azure OpenAI…
✅ Response: OK
🆔 Request ID: chatcmpl-xxxx
📊 Prompt tokens: 5
📊 Completion tokens: 1
📊 Total tokens: 6
⏱️  Latency: 234.56ms
🏁 Finish reason: stop

💬 Response content: OK

✅ Foundry verification complete! Check Azure AI Foundry logs.
   - Log Analytics: Azure Portal > Log Analytics Workspace > Logs
   - Application Insights: Azure Portal > Application Insights > Logs
```

---

## Post-Migration Next Steps

After the verification script succeeds, use this checklist to make sure production telemetry is fully wired up.

### 1. Confirm App Service → Log Analytics

```bash
# Capture App Service resource ID
APP_SERVICE_ID=$(az webapp show \
  --name clinic-ai-backend \
  --resource-group clinic-ai \
  --query id -o tsv)

# List diagnostic settings (should show app-service-logs)
az monitor diagnostic-settings list \
  --resource $APP_SERVICE_ID \
  --output table
```

If nothing is listed, recreate the diagnostic setting:

```bash
WORKSPACE_ID=$(az monitor log-analytics workspace show \
  --resource-group clinic-ai \
  --workspace-name Clinic-ai-logs \
  --query id -o tsv)

az monitor diagnostic-settings create \
  --name app-service-logs \
  --resource $APP_SERVICE_ID \
  --workspace $WORKSPACE_ID \
  --logs '[
      {"category":"AppServiceHTTPLogs","enabled":true},
      {"category":"AppServiceConsoleLogs","enabled":true},
      {"category":"AppServiceAppLogs","enabled":true},
      {"category":"AppServiceAuditLogs","enabled":true},
      {"category":"AppServiceIPSecAuditLogs","enabled":true}
  ]' \
  --metrics '[{"category":"AllMetrics","enabled":true}]'
```

### 2. Validate data in Log Analytics

Run in `Clinic-ai-logs`:

```kql
AppServiceHTTPLogs
| where TimeGenerated > ago(15m)
| where _ResourceId contains "/sites/clinic-ai-backend"
| project TimeGenerated, CsMethod, CsUriStem, CsHost, ScStatus, Clp, UserAgent
| take 20
```

Once you see rows, HTTP diagnostics are flowing.

### 3. Validate Application Insights traces

```kql
AppRequests
| where TimeGenerated > ago(15m)
| where cloud_RoleName == "clinic-ai-backend"
| order by TimeGenerated desc
| take 20
```

If nothing appears, double-check `APPLICATIONINSIGHTS_CONNECTION_STRING` in App Service and restart.

### 4. Clean up duplicate Azure resources

- Remove the extra Foundry workspace (`clinicai-foundry`) if `clinicai-project-resource` is the one you use.
- Remove unused Log Analytics workspaces (`workspace-clinicaifBYY`) so all diagnostics land in `Clinic-ai-logs`.
- Delete any remaining Helicone Container Apps, PostgreSQL, and Container Registry resources.
- Remove unused managed identities (keep only the ones referenced by App Service).

### 5. Create dashboards & alerts

- Build a Log Analytics workbook for token usage, latency, and error rate using the KQL snippets below.
- Create alert rules (5xx, high latency, audit gaps) using `AppRequests` or `AppServiceHTTPLogs`.

### 6. Run an end-to-end workflow

1. Register a test patient via the UI.
2. Upload a visit audio file and wait for transcription.
3. Generate SOAP/summary outputs.
4. Confirm the calls show up in Azure AI Foundry → Monitoring.
5. Confirm the same events appear in `Clinic-ai-logs` and Application Insights.

### Test 2: Validate Logs in Log Analytics

```kql
// Query Azure OpenAI logs
AzureDiagnostics
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| order by TimeGenerated desc
| take 10
| project TimeGenerated, OperationName, DurationMs, ResultType, properties_s
```

### Test 3: Check Application Insights

```kql
// Query app traces
AppTraces
| where AppRoleName == "clinic-ai-backend"
| where TimeGenerated > ago(1h)
| order by TimeGenerated desc
| take 10
```

---

## Monitoring and Observability

### Key Metrics to Monitor

1. **Token Usage**
   - Prompt tokens per request
   - Completion tokens per request
   - Total tokens per day
   - Cost projection

2. **Latency**
   - P50, P95, P99 latencies
   - Time to first token
   - Full request duration

3. **Error Rates**
   - 4xx errors (client errors)
   - 5xx errors (server errors)
   - Timeout errors

4. **Request Volume**
   - Requests per minute
   - Requests by endpoint
   - Requests by deployment

### Azure AI Foundry Dashboard

Navigate to: https://ai.azure.com → **Your workspace** → **Monitoring**

**Views Available**:
- 📊 **Usage**: Token consumption, request counts
- ⏱️ **Latency**: Response time analytics
- ❌ **Errors**: Failure tracking
- 💰 **Cost**: Spending analysis
- 📈 **Trends**: Historical patterns

---

## KQL Query Library

See `AZURE_AI_FOUNDRY_KQL_QUERIES.md` for complete library.

### Quick Queries

#### 1. Today's Token Usage

```kql
AzureDiagnostics
| where TimeGenerated > startofday(now())
| where Category == "RequestResponse"
| extend TokenUsage = toint(properties_s.tokens_total)
| summarize TotalTokens = sum(TokenUsage)
```

#### 2. Average Latency by Hour

```kql
AzureDiagnostics
| where TimeGenerated > ago(24h)
| where Category == "RequestResponse"
| summarize AvgLatency = avg(DurationMs) by bin(TimeGenerated, 1h)
| render timechart
```

#### 3. Error Rate

```kql
AzureDiagnostics
| where TimeGenerated > ago(1h)
| where Category == "RequestResponse"
| summarize 
    Total = count(),
    Errors = countif(ResultType != "Success")
| extend ErrorRate = (Errors * 100.0) / Total
```

---

## Troubleshooting

### Issue 1: Logs Not Appearing in Log Analytics

**Symptoms**: No logs in `AzureDiagnostics` table

**Causes**:
- Diagnostic settings not enabled
- Wrong workspace selected
- Logs delayed (can take 5-10 minutes)

**Solutions**:
1. Verify diagnostic settings are enabled
2. Check Log Analytics workspace is correct
3. Wait 10-15 minutes for initial logs
4. Test with a simple API call

### Issue 2: 404 Deployment Not Found

**Symptoms**: `ValueError: Azure OpenAI deployment not found`

**Causes**:
- Deployment name mismatch
- Wrong API version
- Deployment not in "Succeeded" state

**Solutions**:
1. Verify `AZURE_OPENAI_DEPLOYMENT_NAME` matches Azure Portal
2. Check deployment status is "Succeeded"
3. Try alternative API versions (auto-retry implemented)

### Issue 3: Application Insights Not Connected

**Symptoms**: No traces in Application Insights

**Causes**:
- Connection string not set
- OpenTelemetry not initialized
- Package not installed

**Solutions**:
1. Set `APPLICATIONINSIGHTS_CONNECTION_STRING`
2. Verify packages: `azure-monitor-opentelemetry`, `opentelemetry-instrumentation-fastapi`
3. Restart application

### Issue 4: High Latency

**Symptoms**: Slow API responses

**Causes**:
- Network issues
- Model overload
- Large prompts
- Token throughput limits

**Solutions**:
1. Monitor latency in Foundry dashboard
2. Increase provisioned throughput
3. Optimize prompts (reduce tokens)
4. Enable caching (if applicable)

---

## Cost Optimization

### 1. Token Usage Optimization

**Strategies**:
- ✅ Standardize on `gpt-4o-mini` for every workflow (already provisioned)
- ✅ Reduce `max_tokens` parameter
- ✅ Implement prompt caching
- ✅ Use system messages efficiently

**Example Cost**:
- 1M tokens with gpt-4o-mini: ~$0.75

### 2. Log Retention

**Default**: 30 days (free)
**Extended**: 730 days (compliance)

**Cost**: ~$0.10 per GB/month for retention beyond 30 days

**Recommendation**:
- Keep 30 days for operational logs
- Archive to Azure Storage for long-term retention (cheaper)

### 3. Application Insights Sampling

**Default**: 100% sampling
**Recommended**: 10-20% sampling for high-volume apps

**Configuration**:
```python
from azure.monitor.opentelemetry import configure_azure_monitor

configure_azure_monitor(
    connection_string=conn_str,
    enable_live_metrics=True,
    sampling_ratio=0.2  # 20% sampling
)
```

**Savings**: 80% reduction in App Insights costs

### 4. Provisioned Throughput vs Pay-As-You-Go

**Pay-As-You-Go**:
- Best for: Variable workload
- Cost: $0.15 per 1M input tokens

**Provisioned Throughput**:
- Best for: Predictable, high-volume workload
- Cost: $365/month for 1M tokens/month
- Break-even: ~2.4M tokens/month

**Recommendation**: Start with pay-as-you-go, switch to provisioned when usage is consistent

---

## Best Practices

### 1. Error Handling

✅ **Implement exponential backoff**
```python
max_retries = 3
base_delay = 1.0
for attempt in range(max_retries):
    try:
        response = await client.chat(...)
        break
    except Exception as e:
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            await asyncio.sleep(delay)
```

✅ **Use structured error responses**
```python
return {
    "success": False,
    "error": {
        "code": "RATE_LIMIT_EXCEEDED",
        "message": "Too many requests",
        "details": {"retry_after": 60}
    }
}
```

### 2. Request ID Tracking

✅ **Generate custom request IDs**
```python
import uuid

request_id = f"clinic-ai-{uuid.uuid4()}"
response = await client.chat(
    messages=messages,
    extra_headers={"x-ms-client-request-id": request_id}
)
```

✅ **Log request IDs for correlation**
```python
logger.info(
    "AI request completed",
    extra={
        "request_id": request_id,
        "patient_id": patient_id,
        "tokens": response.usage.total_tokens
    }
)
```

### 3. Token Budget Management

✅ **Set max_tokens appropriately**
```python
# For short responses (questions)
max_tokens = 100

# For medium responses (summaries)
max_tokens = 500

# For long responses (SOAP notes)
max_tokens = 2000
```

✅ **Monitor token usage**
```python
total_tokens = response.usage.total_tokens
if total_tokens > 3000:
    logger.warning(f"High token usage: {total_tokens}")
```

### 4. Security

✅ **Use Key Vault for secrets**
```bash
# Store in Key Vault
az keyvault secret set \
  --vault-name your-key-vault \
  --name AZURE-OPENAI-API-KEY \
  --value "your_api_key"

# Reference in App Service
AZURE_OPENAI_API_KEY=@Microsoft.KeyVault(SecretUri=https://your-keyvault.vault.azure.net/secrets/AZURE-OPENAI-API-KEY/)
```

✅ **Enable managed identity**
```bash
# Enable system-assigned managed identity
az webapp identity assign \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP

# Grant access to Azure OpenAI
az role assignment create \
  --assignee $MANAGED_IDENTITY_ID \
  --role "Cognitive Services OpenAI User" \
  --scope $OPENAI_ID
```

### 5. Monitoring Alerts

✅ **Set up alerts for critical metrics**

```bash
# High error rate alert
az monitor metrics alert create \
  --name high-error-rate \
  --resource-group $RESOURCE_GROUP \
  --scopes $OPENAI_ID \
  --condition "avg RequestErrors > 10" \
  --window-size 5m \
  --evaluation-frequency 1m \
  --action email:your-email@example.com

# High latency alert
az monitor metrics alert create \
  --name high-latency \
  --resource-group $RESOURCE_GROUP \
  --scopes $OPENAI_ID \
  --condition "avg Duration > 5000" \
  --window-size 5m \
  --evaluation-frequency 1m \
  --action email:your-email@example.com
```

---

## Summary

Clinic-AI is now **100% Azure-native** with comprehensive monitoring via Azure AI Foundry:

✅ **Helicone Removed**: No third-party dependencies  
✅ **Zero Latency**: Direct Azure OpenAI calls  
✅ **Full Observability**: Logs, metrics, traces in one place  
✅ **Cost Efficient**: No per-request fees  
✅ **HIPAA Compliant**: PHI stays in Azure  
✅ **Production Ready**: Proven architecture  

For questions or support, contact the Clinic-AI team.

---

**Document Version**: 1.0  
**Last Updated**: November 27, 2025  
**Maintained By**: Clinic-AI DevOps Team

