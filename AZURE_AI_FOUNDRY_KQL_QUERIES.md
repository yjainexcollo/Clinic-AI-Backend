# Azure AI Foundry - KQL Query Library

**Purpose**: Ready-to-use KQL queries for monitoring Clinic-AI AI operations  
**Last Updated**: November 27, 2025

---

## Table of Contents

1. [Token Usage Queries](#token-usage-queries)
2. [Latency Analysis](#latency-analysis)
3. [Error Tracking](#error-tracking)
4. [Cost Analysis](#cost-analysis)
5. [Request Volume](#request-volume)
6. [Performance Optimization](#performance-optimization)
7. [Security Audit](#security-audit)
8. [Troubleshooting](#troubleshooting)

---

## Token Usage Queries

### 1. Total Tokens Today

```kql
AzureDiagnostics
| where TimeGenerated > startofday(now())
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| extend TokensPrompt = toint(parse_json(properties_s).usage.prompt_tokens)
| extend TokensCompletion = toint(parse_json(properties_s).usage.completion_tokens)
| extend TokensTotal = toint(parse_json(properties_s).usage.total_tokens)
| summarize 
    TotalPromptTokens = sum(TokensPrompt),
    TotalCompletionTokens = sum(TokensCompletion),
    TotalTokens = sum(TokensTotal),
    RequestCount = count()
| extend AvgTokensPerRequest = TotalTokens / RequestCount
| project 
    TotalPromptTokens, 
    TotalCompletionTokens, 
    TotalTokens, 
    RequestCount,
    AvgTokensPerRequest
```

### 2. Token Usage by Hour (Last 24h)

```kql
AzureDiagnostics
| where TimeGenerated > ago(24h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| extend TokensTotal = toint(parse_json(properties_s).usage.total_tokens)
| summarize TotalTokens = sum(TokensTotal) by bin(TimeGenerated, 1h)
| render timechart 
    with (
        title="Token Usage per Hour",
        xtitle="Time",
        ytitle="Total Tokens"
    )
```

### 3. Token Usage by Deployment

```kql
AzureDiagnostics
| where TimeGenerated > ago(7d)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| extend Deployment = tostring(parse_json(properties_s).modelDeploymentName)
| extend TokensTotal = toint(parse_json(properties_s).usage.total_tokens)
| summarize 
    TotalTokens = sum(TokensTotal),
    RequestCount = count()
    by Deployment
| order by TotalTokens desc
```

### 4. Top 10 Requests by Token Consumption

```kql
AzureDiagnostics
| where TimeGenerated > ago(1d)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| extend TokensTotal = toint(parse_json(properties_s).usage.total_tokens)
| extend RequestId = tostring(parse_json(properties_s).request_id)
| top 10 by TokensTotal desc
| project TimeGenerated, RequestId, TokensTotal, DurationMs, ResultType
```

### 5. Average Tokens per Request (Last 7 Days)

```kql
AzureDiagnostics
| where TimeGenerated > ago(7d)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| extend TokensPrompt = toint(parse_json(properties_s).usage.prompt_tokens)
| extend TokensCompletion = toint(parse_json(properties_s).usage.completion_tokens)
| summarize 
    AvgPromptTokens = avg(TokensPrompt),
    AvgCompletionTokens = avg(TokensCompletion),
    AvgTotalTokens = avg(TokensPrompt + TokensCompletion)
| project AvgPromptTokens, AvgCompletionTokens, AvgTotalTokens
```

---

## Latency Analysis

### 6. Latency Percentiles (Last 24h)

```kql
AzureDiagnostics
| where TimeGenerated > ago(24h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| summarize 
    P50 = percentile(DurationMs, 50),
    P95 = percentile(DurationMs, 95),
    P99 = percentile(DurationMs, 99),
    Max = max(DurationMs),
    Avg = avg(DurationMs)
| project P50, P95, P99, Max, Avg
```

### 7. Latency Trend by Hour

```kql
AzureDiagnostics
| where TimeGenerated > ago(24h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| summarize 
    AvgLatency = avg(DurationMs),
    P95Latency = percentile(DurationMs, 95)
    by bin(TimeGenerated, 1h)
| render timechart 
    with (
        title="Latency Trend (Avg and P95)",
        xtitle="Time",
        ytitle="Latency (ms)"
    )
```

### 8. Slow Requests (> 5 seconds)

```kql
AzureDiagnostics
| where TimeGenerated > ago(24h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| where DurationMs > 5000
| extend RequestId = tostring(parse_json(properties_s).request_id)
| extend TokensTotal = toint(parse_json(properties_s).usage.total_tokens)
| project TimeGenerated, RequestId, DurationMs, TokensTotal, OperationName
| order by DurationMs desc
```

### 9. Latency by Deployment

```kql
AzureDiagnostics
| where TimeGenerated > ago(24h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| extend Deployment = tostring(parse_json(properties_s).modelDeploymentName)
| summarize 
    AvgLatency = avg(DurationMs),
    P95Latency = percentile(DurationMs, 95),
    RequestCount = count()
    by Deployment
| order by AvgLatency desc
```

### 10. Latency Distribution (Histogram)

```kql
AzureDiagnostics
| where TimeGenerated > ago(24h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| summarize count() by bin(DurationMs, 500)
| render columnchart 
    with (
        title="Latency Distribution",
        xtitle="Latency (ms)",
        ytitle="Request Count"
    )
```

---

## Error Tracking

### 11. Error Rate (Last Hour)

```kql
AzureDiagnostics
| where TimeGenerated > ago(1h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| summarize 
    Total = count(),
    Errors = countif(ResultType != "Success"),
    Success = countif(ResultType == "Success")
| extend ErrorRate = (Errors * 100.0) / Total
| project Total, Success, Errors, ErrorRate
```

### 12. Errors by Type

```kql
AzureDiagnostics
| where TimeGenerated > ago(24h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| where ResultType != "Success"
| extend ErrorCode = tostring(parse_json(properties_s).error.code)
| extend ErrorMessage = tostring(parse_json(properties_s).error.message)
| summarize Count = count() by ErrorCode, ErrorMessage
| order by Count desc
```

### 13. 429 Rate Limit Errors

```kql
AzureDiagnostics
| where TimeGenerated > ago(24h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| where ResultType == "429" or ResultType contains "RateLimitExceeded"
| extend RequestId = tostring(parse_json(properties_s).request_id)
| project TimeGenerated, RequestId, ResultType, OperationName
| order by TimeGenerated desc
```

### 14. Error Trend by Hour

```kql
AzureDiagnostics
| where TimeGenerated > ago(24h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| summarize 
    TotalRequests = count(),
    ErrorCount = countif(ResultType != "Success")
    by bin(TimeGenerated, 1h)
| extend ErrorRate = (ErrorCount * 100.0) / TotalRequests
| render timechart 
    with (
        title="Error Rate Trend",
        xtitle="Time",
        ytitle="Error Rate (%)"
    )
```

### 15. Failed Requests with Details

```kql
AzureDiagnostics
| where TimeGenerated > ago(24h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| where ResultType != "Success"
| extend RequestId = tostring(parse_json(properties_s).request_id)
| extend ErrorCode = tostring(parse_json(properties_s).error.code)
| extend ErrorMessage = tostring(parse_json(properties_s).error.message)
| project 
    TimeGenerated, 
    RequestId, 
    ResultType, 
    ErrorCode, 
    ErrorMessage, 
    DurationMs,
    OperationName
| order by TimeGenerated desc
| take 100
```

---

## Cost Analysis

### 16. Daily Cost Estimate (gpt-4o-mini)

```kql
AzureDiagnostics
| where TimeGenerated > startofday(now())
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| extend TokensPrompt = toint(parse_json(properties_s).usage.prompt_tokens)
| extend TokensCompletion = toint(parse_json(properties_s).usage.completion_tokens)
| summarize 
    TotalPromptTokens = sum(TokensPrompt),
    TotalCompletionTokens = sum(TokensCompletion)
| extend 
    PromptCost = TotalPromptTokens * 0.15 / 1000000,      // $0.15 per 1M input tokens
    CompletionCost = TotalCompletionTokens * 0.60 / 1000000, // $0.60 per 1M output tokens
    TotalCost = (TotalPromptTokens * 0.15 / 1000000) + (TotalCompletionTokens * 0.60 / 1000000)
| project 
    TotalPromptTokens, 
    TotalCompletionTokens, 
    PromptCost = round(PromptCost, 2),
    CompletionCost = round(CompletionCost, 2),
    TotalCost = round(TotalCost, 2)
```

### 17. Cost Trend (Last 7 Days)

```kql
AzureDiagnostics
| where TimeGenerated > ago(7d)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| extend TokensPrompt = toint(parse_json(properties_s).usage.prompt_tokens)
| extend TokensCompletion = toint(parse_json(properties_s).usage.completion_tokens)
| summarize 
    TotalPromptTokens = sum(TokensPrompt),
    TotalCompletionTokens = sum(TokensCompletion)
    by bin(TimeGenerated, 1d)
| extend DailyCost = (TotalPromptTokens * 0.15 / 1000000) + (TotalCompletionTokens * 0.60 / 1000000)
| project TimeGenerated, DailyCost = round(DailyCost, 2)
| render timechart 
    with (
        title="Daily Cost Trend",
        xtitle="Date",
        ytitle="Cost ($)"
    )
```

### 18. Monthly Cost Projection

```kql
AzureDiagnostics
| where TimeGenerated > startofmonth(now())
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| extend TokensPrompt = toint(parse_json(properties_s).usage.prompt_tokens)
| extend TokensCompletion = toint(parse_json(properties_s).usage.completion_tokens)
| summarize 
    TotalPromptTokens = sum(TokensPrompt),
    TotalCompletionTokens = sum(TokensCompletion)
| extend 
    CurrentCost = (TotalPromptTokens * 0.15 / 1000000) + (TotalCompletionTokens * 0.60 / 1000000),
    DaysInMonth = dayofmonth(endofmonth(now())),
    DaysElapsed = dayofmonth(now())
| extend ProjectedMonthlyCost = CurrentCost / DaysElapsed * DaysInMonth
| project 
    CurrentCost = round(CurrentCost, 2),
    ProjectedMonthlyCost = round(ProjectedMonthlyCost, 2),
    DaysElapsed,
    DaysInMonth
```

### 19. Cost by Service (SOAP, Question, Action Plan)

```kql
AppTraces
| where TimeGenerated > ago(7d)
| where Message contains "AI_CALL"
| extend PromptName = extract("prompt_name=([^ ]+)", 1, Message)
| extend Tokens = toint(extract("tokens=([0-9]+)", 1, Message))
| summarize 
    TotalTokens = sum(Tokens),
    RequestCount = count()
    by PromptName
| extend EstimatedCost = TotalTokens * 0.375 / 1000000  // Average cost
| order by EstimatedCost desc
| project PromptName, TotalTokens, RequestCount, EstimatedCost = round(EstimatedCost, 4)
```

### 20. Most Expensive Requests

```kql
AzureDiagnostics
| where TimeGenerated > ago(24h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| extend TokensPrompt = toint(parse_json(properties_s).usage.prompt_tokens)
| extend TokensCompletion = toint(parse_json(properties_s).usage.completion_tokens)
| extend RequestId = tostring(parse_json(properties_s).request_id)
| extend Cost = (TokensPrompt * 0.15 / 1000000) + (TokensCompletion * 0.60 / 1000000)
| top 20 by Cost desc
| project 
    TimeGenerated, 
    RequestId, 
    TokensPrompt, 
    TokensCompletion, 
    Cost = round(Cost, 4),
    DurationMs
```

---

## Request Volume

### 21. Requests per Minute (Last Hour)

```kql
AzureDiagnostics
| where TimeGenerated > ago(1h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| summarize RequestCount = count() by bin(TimeGenerated, 1m)
| render timechart 
    with (
        title="Requests per Minute",
        xtitle="Time",
        ytitle="Request Count"
    )
```

### 22. Request Volume by Operation

```kql
AzureDiagnostics
| where TimeGenerated > ago(24h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| summarize Count = count() by OperationName
| order by Count desc
| render piechart 
    with (title="Request Distribution by Operation")
```

### 23. Peak Traffic Times

```kql
AzureDiagnostics
| where TimeGenerated > ago(7d)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| extend Hour = hourofday(TimeGenerated)
| summarize AvgRequestsPerHour = count() / 7.0 by Hour
| order by AvgRequestsPerHour desc
| render columnchart 
    with (
        title="Average Requests by Hour of Day",
        xtitle="Hour (0-23)",
        ytitle="Avg Requests"
    )
```

### 24. Request Growth Trend

```kql
AzureDiagnostics
| where TimeGenerated > ago(30d)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| summarize DailyRequests = count() by bin(TimeGenerated, 1d)
| order by TimeGenerated asc
| render timechart 
    with (
        title="Request Growth (30 Days)",
        xtitle="Date",
        ytitle="Daily Requests"
    )
```

### 25. Concurrent Requests

```kql
AzureDiagnostics
| where TimeGenerated > ago(1h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| extend EndTime = TimeGenerated + totimespan(DurationMs * 1ms)
| summarize ConcurrentRequests = count() by bin(TimeGenerated, 1s)
| summarize 
    MaxConcurrent = max(ConcurrentRequests),
    AvgConcurrent = avg(ConcurrentRequests)
```

---

## Performance Optimization

### 26. Identify Large Prompts (> 2000 tokens)

```kql
AzureDiagnostics
| where TimeGenerated > ago(24h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| extend TokensPrompt = toint(parse_json(properties_s).usage.prompt_tokens)
| where TokensPrompt > 2000
| extend RequestId = tostring(parse_json(properties_s).request_id)
| project 
    TimeGenerated, 
    RequestId, 
    TokensPrompt, 
    DurationMs, 
    OperationName
| order by TokensPrompt desc
```

### 27. Requests with High Completion Tokens

```kql
AzureDiagnostics
| where TimeGenerated > ago(24h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| extend TokensCompletion = toint(parse_json(properties_s).usage.completion_tokens)
| where TokensCompletion > 1000
| extend RequestId = tostring(parse_json(properties_s).request_id)
| project 
    TimeGenerated, 
    RequestId, 
    TokensCompletion, 
    DurationMs, 
    OperationName
| order by TokensCompletion desc
```

### 28. Correlation: Tokens vs Latency

```kql
AzureDiagnostics
| where TimeGenerated > ago(24h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| extend TokensTotal = toint(parse_json(properties_s).usage.total_tokens)
| project TokensTotal, DurationMs
| render scatterchart 
    with (
        title="Tokens vs Latency Correlation",
        xtitle="Total Tokens",
        ytitle="Latency (ms)"
    )
```

### 29. Finish Reason Analysis

```kql
AzureDiagnostics
| where TimeGenerated > ago(24h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| extend FinishReason = tostring(parse_json(properties_s).choices[0].finish_reason)
| summarize Count = count() by FinishReason
| render piechart 
    with (title="Distribution of Finish Reasons")
```

### 30. Requests Hitting max_tokens

```kql
AzureDiagnostics
| where TimeGenerated > ago(24h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| extend FinishReason = tostring(parse_json(properties_s).choices[0].finish_reason)
| where FinishReason == "length"
| extend TokensCompletion = toint(parse_json(properties_s).usage.completion_tokens)
| extend RequestId = tostring(parse_json(properties_s).request_id)
| project 
    TimeGenerated, 
    RequestId, 
    TokensCompletion, 
    OperationName
| order by TimeGenerated desc
```

---

## Security Audit

### 31. Requests by IP Address

```kql
AppRequests
| where TimeGenerated > ago(24h)
| where Url contains "openai"
| summarize RequestCount = count() by ClientIP
| order by RequestCount desc
| take 20
```

### 32. Unauthorized Access Attempts

```kql
AzureDiagnostics
| where TimeGenerated > ago(24h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| where ResultType == "401" or ResultType == "403"
| extend RequestId = tostring(parse_json(properties_s).request_id)
| project TimeGenerated, RequestId, ResultType, CallerIPAddress, OperationName
| order by TimeGenerated desc
```

### 33. API Key Usage Pattern

```kql
AzureDiagnostics
| where TimeGenerated > ago(7d)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| extend ApiKeyPrefix = substring(tostring(parse_json(properties_s).api_key), 0, 8)
| summarize RequestCount = count() by ApiKeyPrefix, bin(TimeGenerated, 1d)
| render timechart 
    with (
        title="API Key Usage Pattern",
        xtitle="Date",
        ytitle="Request Count"
    )
```

### 34. Suspicious Activity Detection

```kql
AzureDiagnostics
| where TimeGenerated > ago(1h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| summarize 
    RequestCount = count(),
    ErrorCount = countif(ResultType != "Success"),
    AvgLatency = avg(DurationMs)
    by CallerIPAddress
| where RequestCount > 100 or ErrorCount > 10 or AvgLatency > 10000
| order by RequestCount desc
```

### 35. Audit Trail

```kql
AzureDiagnostics
| where TimeGenerated > ago(24h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "Audit"
| extend RequestId = tostring(parse_json(properties_s).request_id)
| extend UserId = tostring(parse_json(properties_s).user_id)
| project 
    TimeGenerated, 
    OperationName, 
    RequestId, 
    UserId, 
    ResultType, 
    CallerIPAddress
| order by TimeGenerated desc
```

---

## Troubleshooting

### 36. Recent Errors with Stack Traces

```kql
AppExceptions
| where TimeGenerated > ago(1h)
| where AppRoleName == "clinic-ai-backend"
| project 
    TimeGenerated, 
    ExceptionType, 
    OuterMessage, 
    InnermostMessage, 
    Details
| order by TimeGenerated desc
| take 20
```

### 37. Request Correlation (Full Journey)

```kql
let requestId = "your-request-id-here";
union
    (AzureDiagnostics
    | where TimeGenerated > ago(1h)
    | where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
    | extend RequestId = tostring(parse_json(properties_s).request_id)
    | where RequestId == requestId
    | extend Source = "Azure OpenAI"
    | project TimeGenerated, Source, OperationName, ResultType, DurationMs, properties_s),
    (AppRequests
    | where TimeGenerated > ago(1h)
    | where Properties.RequestId == requestId
    | extend Source = "App Request"
    | project TimeGenerated, Source, Name, ResultCode, DurationMs = Duration, Properties),
    (AppTraces
    | where TimeGenerated > ago(1h)
    | where Properties.RequestId == requestId
    | extend Source = "App Trace"
    | project TimeGenerated, Source, Message, SeverityLevel, Properties)
| order by TimeGenerated asc
```

### 38. Deployment Health Check

```kql
AzureDiagnostics
| where TimeGenerated > ago(5m)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| extend Deployment = tostring(parse_json(properties_s).modelDeploymentName)
| summarize 
    TotalRequests = count(),
    SuccessfulRequests = countif(ResultType == "Success"),
    FailedRequests = countif(ResultType != "Success"),
    AvgLatency = avg(DurationMs)
    by Deployment
| extend 
    SuccessRate = (SuccessfulRequests * 100.0) / TotalRequests,
    Health = case(
        SuccessRate >= 99.0 and AvgLatency < 3000, "Healthy",
        SuccessRate >= 95.0 and AvgLatency < 5000, "Degraded",
        "Unhealthy"
    )
| project Deployment, Health, SuccessRate, AvgLatency, TotalRequests
```

### 39. Missing Logs Investigation

```kql
// Check if diagnostic settings are enabled
AzureDiagnostics
| where TimeGenerated > ago(1h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| summarize 
    LogCount = count(),
    LatestLog = max(TimeGenerated),
    EarliestLog = min(TimeGenerated)
| extend 
    TimeSinceLastLog = now() - LatestLog,
    Status = case(
        LogCount == 0, "No logs found - check diagnostic settings",
        TimeSinceLastLog > 15m, "Logs delayed - investigate",
        "Logs flowing normally"
    )
| project Status, LogCount, LatestLog, TimeSinceLastLog
```

### 40. Performance Degradation Detection

```kql
let baselineLatency = toscalar(
    AzureDiagnostics
    | where TimeGenerated between (ago(7d) .. ago(1h))
    | where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
    | where Category == "RequestResponse"
    | summarize avg(DurationMs)
);
AzureDiagnostics
| where TimeGenerated > ago(1h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| summarize CurrentLatency = avg(DurationMs)
| extend 
    BaselineLatency = baselineLatency,
    LatencyIncrease = ((CurrentLatency - baselineLatency) / baselineLatency) * 100,
    Status = case(
        LatencyIncrease > 50, "🔴 Critical degradation",
        LatencyIncrease > 20, "🟡 Moderate degradation",
        "🟢 Normal"
    )
| project Status, CurrentLatency, BaselineLatency, LatencyIncrease
```

---

## Custom Alerts (Examples)

### Alert 1: High Error Rate

```kql
// Trigger when error rate > 5% in last 5 minutes
AzureDiagnostics
| where TimeGenerated > ago(5m)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| summarize 
    Total = count(),
    Errors = countif(ResultType != "Success")
| extend ErrorRate = (Errors * 100.0) / Total
| where ErrorRate > 5.0
```

### Alert 2: High Latency

```kql
// Trigger when P95 latency > 5000ms in last 10 minutes
AzureDiagnostics
| where TimeGenerated > ago(10m)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| summarize P95Latency = percentile(DurationMs, 95)
| where P95Latency > 5000
```

### Alert 3: Unusual Token Consumption

```kql
// Trigger when hourly tokens > 2x daily average
let dailyAvg = toscalar(
    AzureDiagnostics
    | where TimeGenerated > ago(7d)
    | where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
    | where Category == "RequestResponse"
    | extend TokensTotal = toint(parse_json(properties_s).usage.total_tokens)
    | summarize avg(TokensTotal) * 24.0
);
AzureDiagnostics
| where TimeGenerated > ago(1h)
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where Category == "RequestResponse"
| extend TokensTotal = toint(parse_json(properties_s).usage.total_tokens)
| summarize HourlyTokens = sum(TokensTotal)
| where HourlyTokens > dailyAvg * 2
```

---

## Tips for Using KQL

1. **Use `ago()` for relative time ranges**: `ago(24h)`, `ago(7d)`, `ago(1h)`
2. **Filter early**: Place `where` clauses as early as possible for performance
3. **Use `extend` for calculated fields**: `extend Cost = Tokens * 0.15 / 1000000`
4. **Aggregate with `summarize`**: `summarize count() by Category`
5. **Visualize with `render`**: `render timechart`, `render piechart`
6. **Parse JSON**: `parse_json(properties_s).usage.total_tokens`
7. **Save frequent queries**: Use "Save" in Log Analytics for quick access

---

**For more KQL documentation**: https://learn.microsoft.com/en-us/azure/data-explorer/kusto/query/

**Maintained By**: Clinic-AI DevOps Team  
**Last Updated**: November 27, 2025

