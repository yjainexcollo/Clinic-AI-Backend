# Azure Application Insights KQL Queries for Prompt Versions

This document contains KQL (Kusto Query Language) queries for monitoring and analyzing prompt versions in Azure Application Insights.

## Overview

Prompt versions are automatically tracked in Application Insights through the `llm_gateway.py` module. Each LLM call includes:
- `llm.prompt_version`: The version string (e.g., "1.3", "2.0")
- `llm.scenario`: The prompt scenario (intake, soap_summary, previsit_summary, postvisit_summary)
- `llm.latency_ms`: Response latency in milliseconds
- `llm.tokens`: Token usage
- Success/failure status

## Table of Contents

1. [Current Version Queries](#current-version-queries)
2. [Version Usage Analysis](#version-usage-analysis)
3. [Performance Comparison](#performance-comparison)
4. [Version History Tracking](#version-history-tracking)
5. [Error Analysis](#error-analysis)
6. [Dashboard Queries](#dashboard-queries)
7. [Alert Queries](#alert-queries)

---

## Current Version Queries

### Query 1: Current Version for All Scenarios

**Purpose**: Get the current active version for each prompt scenario based on the most recent telemetry.

**When to Use**: 
- Quick check of what versions are currently deployed
- Daily monitoring
- After deployments to verify new versions are active

```kql
traces
| where name == "llm_call"
    and timestamp > ago(24h)
| extend prompt_version = tostring(customDimensions["llm.prompt_version"])
| extend scenario = tostring(customDimensions["llm.scenario"])
| summarize 
    current_version = take_any(prompt_version),
    last_used = max(timestamp),
    calls_count = count()
    by scenario
| extend last_used_ago = datetime_diff('hour', now(), last_used)
| project 
    scenario, 
    current_version, 
    calls_last_24h = calls_count,
    last_used,
    hours_ago = last_used_ago
| order by scenario
```

**Output Columns**:
- `scenario`: Prompt scenario name
- `current_version`: Currently active version (e.g., "1.3")
- `calls_last_24h`: Number of calls in last 24 hours
- `last_used`: Timestamp of last use
- `hours_ago`: Hours since last use

---

### Query 2: Current Version for Specific Scenario

**Purpose**: Get the current version for a single scenario (useful for focused checks).

**When to Use**:
- Checking a specific scenario after prompt changes
- Troubleshooting issues with one prompt type
- Validating deployment for specific scenario

```kql
traces
| where name == "llm_call"
| extend prompt_version = tostring(customDimensions["llm.prompt_version"])
| extend scenario = tostring(customDimensions["llm.scenario"])
| where scenario == "soap_summary"  // Change to: intake, previsit_summary, postvisit_summary
| top 1 by timestamp desc
| project scenario, current_version = prompt_version, last_used = timestamp
```

**Modify for other scenarios**:
- `scenario == "intake"` - For intake prompts
- `scenario == "previsit_summary"` - For pre-visit summary prompts
- `scenario == "postvisit_summary"` - For post-visit summary prompts

---

## Version Usage Analysis

### Query 3: Version Distribution by Scenario

**Purpose**: See all versions being used and their distribution across scenarios.

**When to Use**:
- Understanding version adoption
- Identifying if multiple versions are running simultaneously
- Before cleanup/removal of old versions

```kql
traces
| where name == "llm_call"
    and timestamp > ago(7d)
| extend prompt_version = tostring(customDimensions["llm.prompt_version"])
| extend scenario = tostring(customDimensions["llm.scenario"])
| summarize 
    count() by scenario, prompt_version
| order by scenario, prompt_version
```

**Output**: Shows count of calls per version per scenario over last 7 days.

---

### Query 4: Version Transition Timeline

**Purpose**: Track when versions changed over time (identify deployment dates).

**When to Use**:
- Understanding version history
- Correlating deployments with performance changes
- Audit trail for prompt changes

```kql
traces
| where name == "llm_call"
| extend prompt_version = tostring(customDimensions["llm.prompt_version"])
| extend scenario = tostring(customDimensions["llm.scenario"])
| summarize 
    first_seen = min(timestamp),
    last_seen = max(timestamp),
    total_calls = count()
    by scenario, prompt_version
| extend 
    first_seen_date = format_datetime(first_seen, "yyyy-MM-dd HH:mm"),
    last_seen_date = format_datetime(last_seen, "yyyy-MM-dd HH:mm"),
    duration_hours = datetime_diff('hour', last_seen, first_seen)
| project 
    scenario, 
    prompt_version,
    first_seen_date,
    last_seen_date,
    duration_hours,
    total_calls
| order by scenario, first_seen desc
```

---

## Performance Comparison

### Query 5: Performance Metrics by Version

**Purpose**: Compare latency and token usage across different prompt versions.

**When to Use**:
- Evaluating if new versions improve performance
- Identifying performance regressions
- A/B testing analysis

```kql
traces
| where name == "llm_call"
    and timestamp > ago(7d)
| extend prompt_version = tostring(customDimensions["llm.prompt_version"])
| extend scenario = tostring(customDimensions["llm.scenario"])
| extend latency_ms = toreal(customDimensions["llm.latency_ms"])
| extend tokens = toint(customDimensions["llm.tokens"])
| summarize 
    avg_latency_ms = avg(latency_ms),
    min_latency_ms = min(latency_ms),
    max_latency_ms = max(latency_ms),
    p50_latency_ms = percentile(latency_ms, 50),
    p95_latency_ms = percentile(latency_ms, 95),
    p99_latency_ms = percentile(latency_ms, 99),
    avg_tokens = avg(tokens),
    total_tokens = sum(tokens),
    call_count = count()
    by scenario, prompt_version
| order by scenario, prompt_version
```

**Key Metrics**:
- `avg_latency_ms`: Average response time
- `p95_latency_ms`: 95th percentile (catches slow outliers)
- `avg_tokens`: Average token usage per call
- `total_tokens`: Total tokens consumed (for cost analysis)

---

### Query 6: Performance Comparison: Old vs New Version

**Purpose**: Directly compare two specific versions side-by-side.

**When to Use**:
- Before/after deployment comparison
- Validating performance improvements
- Rollback decision support

```kql
traces
| where name == "llm_call"
    and timestamp > ago(7d)
| extend prompt_version = tostring(customDimensions["llm.prompt_version"])
| extend scenario = tostring(customDimensions["llm.scenario"])
| extend latency_ms = toreal(customDimensions["llm.latency_ms"])
| where scenario == "soap_summary"
    and (prompt_version == "1.3" or prompt_version == "1.4")  // Change versions as needed
| summarize 
    avg_latency = avg(latency_ms),
    p95_latency = percentile(latency_ms, 95),
    call_count = count(),
    success_rate = 100.0 * countif(success == true) / count()
    by prompt_version
| order by prompt_version
```

---

## Version History Tracking

### Query 7: Version Usage Over Time (Time Series)

**Purpose**: Visualize version adoption over time (useful for line charts).

**When to Use**:
- Creating dashboards with time-series charts
- Understanding gradual version rollout
- Identifying version rollback events

```kql
traces
| where name == "llm_call"
    and timestamp > ago(30d)
| extend prompt_version = tostring(customDimensions["llm.prompt_version"])
| extend scenario = tostring(customDimensions["llm.scenario"])
| where scenario == "intake"  // Change scenario as needed
| summarize count() by bin(timestamp, 1h), prompt_version
| order by timestamp asc, prompt_version
```

**Visualization**: Use as line chart with timestamp on X-axis, count on Y-axis, and separate lines per version.

---

### Query 8: All Versions Seen for Each Scenario

**Purpose**: Get complete list of all versions that have been used.

**When to Use**:
- Audit purposes
- Understanding version history
- Cleanup planning

```kql
traces
| where name == "llm_call"
| extend prompt_version = tostring(customDimensions["llm.prompt_version"])
| extend scenario = tostring(customDimensions["llm.scenario"])
| summarize 
    all_versions = make_set(prompt_version),
    version_count = dcount(prompt_version),
    first_seen = min(timestamp),
    last_seen = max(timestamp)
    by scenario
| extend versions_list = strcat_array(all_versions, ", ")
| project scenario, versions_list, version_count, first_seen, last_seen
| order by scenario
```

---

## Error Analysis

### Query 9: Error Rate by Version

**Purpose**: Identify if specific versions have higher error rates.

**When to Use**:
- Troubleshooting errors after version changes
- Quality assurance after deployments
- Rollback decision support

```kql
traces
| where name == "llm_call"
    and timestamp > ago(7d)
| extend prompt_version = tostring(customDimensions["llm.prompt_version"])
| extend scenario = tostring(customDimensions["llm.scenario"])
| summarize 
    total_calls = count(),
    error_count = countif(success == false),
    success_count = countif(success == true)
    by scenario, prompt_version
| extend 
    error_rate = 100.0 * error_count / total_calls,
    success_rate = 100.0 * success_count / total_calls
| project 
    scenario, 
    prompt_version, 
    total_calls,
    error_count,
    success_count,
    error_rate,
    success_rate
| order by scenario, error_rate desc
```

---

### Query 10: Recent Errors by Version

**Purpose**: Find recent errors and which versions they occurred with.

**When to Use**:
- Investigating production issues
- Post-deployment monitoring
- Incident response

```kql
traces
| where name == "llm_call"
    and success == false
    and timestamp > ago(24h)
| extend prompt_version = tostring(customDimensions["llm.prompt_version"])
| extend scenario = tostring(customDimensions["llm.scenario"])
| extend error_message = tostring(customDimensions["llm.error"])
| project 
    timestamp,
    scenario,
    prompt_version,
    error_message,
    duration,
    message
| order by timestamp desc
| take 100
```

---

## Dashboard Queries

### Query 11: Dashboard Summary - All Scenarios

**Purpose**: Comprehensive overview for dashboards showing current status of all scenarios.

**When to Use**:
- Main dashboard view
- Executive summary
- Daily health checks

```kql
traces
| where name == "llm_call"
    and timestamp > ago(7d)
| extend prompt_version = tostring(customDimensions["llm.prompt_version"])
| extend scenario = tostring(customDimensions["llm.scenario"])
| extend latency_ms = toreal(customDimensions["llm.latency_ms"])
| summarize 
    current_version = take_any(prompt_version),
    total_calls_7d = count(),
    calls_last_24h = countif(timestamp > ago(24h)),
    avg_latency_ms = avg(latency_ms),
    p95_latency_ms = percentile(latency_ms, 95),
    error_rate = 100.0 * countif(success == false) / count(),
    last_used = max(timestamp)
    by scenario
| extend 
    status = case(
        last_used > ago(1h), "Active",
        last_used > ago(24h), "Recent",
        "Inactive"
    ),
    hours_since_last_use = datetime_diff('hour', now(), last_used)
| project 
    scenario,
    current_version,
    status,
    total_calls_7d,
    calls_last_24h,
    avg_latency_ms,
    p95_latency_ms,
    error_rate,
    hours_since_last_use,
    last_used
| order by scenario
```

**Dashboard Visualization**:
- Use as table view
- Color-code status column (green=Active, yellow=Recent, red=Inactive)
- Create separate charts for latency, error rate, and call volume

---

### Query 12: Version Health Status

**Purpose**: Quick health check showing if versions are working correctly.

**When to Use**:
- Status page
- Health monitoring alerts
- Quick operational checks

```kql
traces
| where name == "llm_call"
    and timestamp > ago(1h)
| extend prompt_version = tostring(customDimensions["llm.prompt_version"])
| extend scenario = tostring(customDimensions["llm.scenario"])
| summarize 
    current_version = take_any(prompt_version),
    call_count = count(),
    error_count = countif(success == false),
    has_errors = countif(success == false) > 0
    by scenario
| extend health = case(
    call_count == 0, "No Activity",
    error_count == 0, "Healthy",
    error_count < call_count * 0.1, "Degraded",
    "Unhealthy"
)
| project scenario, current_version, call_count, error_count, health
| order by scenario
```

---

## Alert Queries

### Query 13: Multiple Versions Detected (Alert)

**Purpose**: Alert when multiple versions are running simultaneously (indicates deployment issue).

**When to Use**:
- Set up alert rule
- Post-deployment validation
- Detecting version rollback issues

```kql
traces
| where name == "llm_call"
    and timestamp > ago(1h)
| extend prompt_version = tostring(customDimensions["llm.prompt_version"])
| extend scenario = tostring(customDimensions["llm.scenario"])
| summarize 
    versions = make_set(prompt_version),
    version_count = dcount(prompt_version)
    by scenario
| where version_count > 1
| extend versions_list = strcat_array(versions, ", ")
| project scenario, version_count, versions_list, alert = "Multiple versions detected"
```

**Alert Configuration**:
- **Alert Condition**: When result count > 0
- **Severity**: Warning
- **Action**: Notify team, check deployment status

---

### Query 14: High Error Rate Alert

**Purpose**: Alert when a version has unusually high error rate.

**When to Use**:
- Set up alert rule for production monitoring
- Early warning system for prompt issues

```kql
traces
| where name == "llm_call"
    and timestamp > ago(15m)
| extend prompt_version = tostring(customDimensions["llm.prompt_version"])
| extend scenario = tostring(customDimensions["llm.scenario"])
| summarize 
    total_calls = count(),
    error_count = countif(success == false)
    by scenario, prompt_version
| where total_calls >= 10  // Minimum calls for statistical significance
| extend error_rate = 100.0 * error_count / total_calls
| where error_rate > 10.0  // Alert if error rate > 10%
| project scenario, prompt_version, total_calls, error_count, error_rate, alert = "High error rate detected"
```

**Alert Configuration**:
- **Alert Condition**: When result count > 0
- **Severity**: Critical if error_rate > 20%, Warning if 10-20%
- **Action**: Notify on-call, investigate errors

---

### Query 15: Version Not Seen Recently (Stale Version)

**Purpose**: Alert if a version that should be active hasn't been seen recently.

**When to Use**:
- Detect if deployment failed
- Verify new versions are actually being used
- Identify stale deployments

```kql
let expected_versions = datatable(scenario:string, expected_version:string)
[
    "intake", "1.4",
    "soap_summary", "1.3",
    "previsit_summary", "1.2",
    "postvisit_summary", "1.1"
];
traces
| where name == "llm_call"
    and timestamp > ago(1h)
| extend prompt_version = tostring(customDimensions["llm.prompt_version"])
| extend scenario = tostring(customDimensions["llm.scenario"])
| summarize current_version = take_any(prompt_version), last_seen = max(timestamp) by scenario
| join kind=leftouter expected_versions on scenario
| where current_version != expected_version or datetime_diff('minute', now(), last_seen) > 60
| project scenario, expected_version, current_version, last_seen, alert = "Version mismatch or stale"
```

**Note**: Update `expected_versions` table with your current expected versions.

---

## Tips for Using These Queries

### 1. Time Range Selection
- Use `ago(24h)` for current/daily checks
- Use `ago(7d)` for weekly analysis
- Use `ago(30d)` for monthly trends

### 2. Scenario Filtering
Replace `scenario == "soap_summary"` with:
- `"intake"` - Patient intake questions
- `"soap_summary"` - SOAP note generation
- `"previsit_summary"` - Pre-visit summary
- `"postvisit_summary"` - Post-visit summary

### 3. Performance Optimization
For large datasets, add time filters early:
```kql
traces
| where timestamp > ago(7d)  // Filter early
| where name == "llm_call"   // Then filter by name
```

### 4. Creating Custom Dashboards
1. Pin Query 11 (Dashboard Summary) as main table
2. Create separate charts for:
   - Latency trends (Query 7 with line chart)
   - Error rates (Query 9 with bar chart)
   - Version distribution (Query 3 with pie chart)

### 5. Setting Up Alerts
1. Use Query 13 for version conflicts
2. Use Query 14 for error rate monitoring
3. Set alert frequency: Every 5-15 minutes
4. Configure action groups for notifications

---

## Version Number Format

Prompt versions follow semantic versioning:
- **Format**: `MAJOR.MINOR` (e.g., "1.3", "2.0")
- **Major Version**: Manually incremented for breaking changes
- **Minor Version**: Automatically incremented when prompt template changes

Examples:
- `1.0` → `1.1` → `1.2` (automatic minor increments)
- `1.4` → `2.0` (manual major increment)
- `2.0` → `2.1` → `2.2` (continues in new major series)

---

## Troubleshooting

### No Results Returned
- Check time range (use `ago(24h)` or larger)
- Verify Application Insights connection string is configured
- Check if LLM calls are actually being made

### "UNKNOWN" Version
- Indicates version was not found in PROMPT_VERSIONS dict
- Usually means startup initialization failed
- Check application logs for version manager errors

### Multiple Versions Detected
- Normal during deployment windows
- Should resolve once old requests complete
- If persistent, check deployment/rollback status

---

## Additional Resources

- [Application Insights Documentation](https://docs.microsoft.com/azure/azure-monitor/app/app-insights-overview)
- [KQL Language Reference](https://docs.microsoft.com/azure/data-explorer/kusto/query/)
- [Prompt Version Manager Code](../src/clinicai/adapters/external/prompt_version_manager.py)

